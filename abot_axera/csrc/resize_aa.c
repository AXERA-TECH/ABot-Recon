/* Antialiased bicubic resize for float32 CHW images (and uint8 HWC input converted on the fly).
 *
 * Mirrors PyTorch's CPU kernel (aten/src/ATen/native/cpu/UpSampleKernel.cpp,
 * _upsample_bicubic2d_aa: HelperInterpCubic::aa_filter + _compute_indices_min_size_weights_aa +
 * interpolate_aa_single_dim, width pass then height pass) expression for expression. Every output
 * element is accumulated over its taps in the same order with the same (fused) multiply-adds, so
 * when compiled with the same contraction rules (-O3 -mfma on x86 / aarch64 default) the output is
 * bit-identical to torchvision.transforms.functional.resize(..., BICUBIC, antialias=True) applied to
 * to_tensor(PIL image) (uint8 -> float32 / 255).
 *
 * Two loop layouts, same math:
 *   default (x86 reference): the per-element scalar tap loop exactly as in aten. gcc -O3 turns it
 *     into the same in-order vectorized reduction that PyTorch's own AVX2 build has, so the result
 *     is bit-identical to torchvision on x86 (verified: 0 differing pixels).
 *   -DABOT_RESIZE_FAST (aarch64 boards): interleaved (R,G,B,0) 4-lane layout so both passes are
 *     contiguous NEON multiply-adds. Same per-element tap order, fused multiply-adds throughout;
 *     differs from the reference layout by <= 1 ulp on part of the pixels. (torch is not a
 *     reference on the boards, so nothing is lost there.)
 *
 * Build: gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC resize_aa.c -o libresize_aa.so -lm            (x86)
 *        gcc -O3 -fopenmp -DABOT_RESIZE_FAST -shared -fPIC resize_aa.c -o libresize_aa.so -lm      (aarch64)
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RB 8   /* rows converted together in the reference-layout uint8 path */

static inline float cubic_convolution1(float x, float A) { return ((A + 2) * x - (A + 3)) * x * x + 1; }
static inline float cubic_convolution2(float x, float A) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; }

static inline float aa_filter(float x) {
    const float a = -0.5f;
    x = fabsf(x);
    if (x < 1.0) return cubic_convolution1(x, a);
    if (x < 2.0) return cubic_convolution2(x, a);
    return 0.0;
}

/* weights for one output index i; returns xmin/xsize; wt has max_interp_size slots */
static inline void weights_aa(int64_t i, int64_t input_size, float scale, float support, float *wt,
                              int64_t max_interp_size, int64_t *xmin, int64_t *xsize) {
    float center = scale * (i + 0.5);
    float total_w = 0.0;
    float invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    int64_t mn = (int64_t)(center - support + 0.5);
    if (mn < 0) mn = 0;
    int64_t mx = (int64_t)(center + support + 0.5);
    if (mx > input_size) mx = input_size;
    int64_t sz = mx - mn;
    if (sz < 0) sz = 0;
    if (sz > max_interp_size) sz = max_interp_size;
    int64_t j = 0;
    for (; j < sz; j++) {
        float w = aa_filter((j + mn - center + 0.5) * invscale);
        wt[j] = w;
        total_w += w;
    }
    if (total_w != 0.0) {
        for (j = 0; j < sz; j++) wt[j] /= total_w;
    }
    for (j = sz; j < max_interp_size; j++) wt[j] = 0.0f;
    *xmin = mn;
    *xsize = sz;
}

typedef struct {
    float *wts;
    int64_t *xmin, *xsize;
    int64_t K;   /* max_interp_size */
} taps_t;

static taps_t make_taps(int64_t in_n, int64_t out_n) {
    const int interp_size = 4;
    float scale = (float)in_n / out_n;
    float support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
    taps_t t;
    t.K = (int64_t)ceil(support) * 2 + 1;
    t.wts = (float *)malloc(sizeof(float) * out_n * t.K);
    t.xmin = (int64_t *)malloc(sizeof(int64_t) * out_n);
    t.xsize = (int64_t *)malloc(sizeof(int64_t) * out_n);
    for (int64_t i = 0; i < out_n; i++)
        weights_aa(i, in_n, scale, support, t.wts + i * t.K, t.K, &t.xmin[i], &t.xsize[i]);
    return t;
}

static void free_taps(taps_t *t) { free(t->wts); free(t->xmin); free(t->xsize); }

#ifndef ABOT_RESIZE_FAST
/* ---- reference layout: aten's interpolate_aa_single_dim, one line at a time ------------------ */
static inline void interp_line_f32(const float *src, float *dst, const taps_t *t, int64_t out_n,
                                   int64_t estride, int64_t ostride) {
    for (int64_t i = 0; i < out_n; i++) {
        const float *s = src + t->xmin[i] * estride;
        const float *w = t->wts + i * t->K;
        int64_t n = t->xsize[i];
        float v = s[0];
        float output = v * w[0];
        for (int64_t j = 1; j < n; j++) {
            float ww = w[j];
            v = s[j * estride];
            output += v * ww;
        }
        dst[i * ostride] = output;
    }
}

/* nr rows (stride rs) -> nr rows of OW (stride OW) */
static inline void wpass_rows(const float *src, int64_t rs, int64_t nr, float *dst, int64_t OW, const taps_t *t) {
    for (int64_t r = 0; r < nr; r++)
        interp_line_f32(src + r * rs, dst + r * OW, t, OW, 1, 1);
}

/* [H, OW] plane -> [OH, OW]: one column at a time (element stride OW) */
static void hpass_plane(const float *tmp, float *out, int64_t H, int64_t OH, int64_t OW, const taps_t *t) {
#pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < OW; x++)
        interp_line_f32(tmp + x, out + x, t, OH, OW, OW);
}
#else
/* ---- interleaved SIMD layout (aarch64) -------------------------------------------------------
 * Pixels are handled as 4-lane groups (R,G,B,0): the width pass computes one output pixel's four
 * lanes per multiply-add (contiguous 16-byte loads), the height pass streams over a contiguous
 * [OW*4] row. Same per-element tap order as the reference layout, fused multiply-adds. */
#define LANES 4

/* one image row in [W][4] float -> [OW][4] float */
static inline void wpass_row4(const float *row4, float *dst4, int64_t OW, const taps_t *t) {
    for (int64_t i = 0; i < OW; i++) {
        const float *s = row4 + t->xmin[i] * LANES;
        const float *w = t->wts + i * t->K;
        const int64_t n = t->xsize[i];
        float acc[LANES];
        const float w0 = w[0];
        for (int l = 0; l < LANES; l++) acc[l] = s[l] * w0;
        for (int64_t j = 1; j < n; j++) {
            const float ww = w[j];
            const float *sj = s + j * LANES;
            for (int l = 0; l < LANES; l++) acc[l] += sj[l] * ww;
        }
        float *d = dst4 + i * LANES;
        for (int l = 0; l < LANES; l++) d[l] = acc[l];
    }
}

/* tmp [H][OW*4] -> out CHW [3][OH][OW] */
static void hpass_interleaved(const float *tmp, float *out, int64_t H, int64_t OH, int64_t OW, const taps_t *t) {
    const int64_t RW = OW * LANES;
#pragma omp parallel
    {
        float *d = (float *)malloc(sizeof(float) * RW);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < OH; i++) {
            const float *w = t->wts + i * t->K;
            const int64_t n = t->xsize[i];
            const float *s0 = tmp + t->xmin[i] * RW;
            const float w0 = w[0];
            for (int64_t x = 0; x < RW; x++) d[x] = s0[x] * w0;
            for (int64_t j = 1; j < n; j++) {
                const float ww = w[j];
                const float *s = s0 + j * RW;
                for (int64_t x = 0; x < RW; x++) d[x] += s[x] * ww;
            }
            for (int c = 0; c < 3; c++) {
                float *o = out + (c * OH + i) * OW;
                for (int64_t x = 0; x < OW; x++) o[x] = d[x * LANES + c];
            }
        }
        free(d);
    }
}

static int resize_interleaved(const float *inf, const uint8_t *inu8, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out) {
    float *tmp = (float *)malloc(sizeof(float) * H * OW * LANES);
    if (!tmp) return -1;
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = (float)i / 255.0f;
    taps_t tw = make_taps(W, OW), th = make_taps(H, OH);
#pragma omp parallel
    {
        float *row4 = (float *)malloc(sizeof(float) * W * LANES);
#pragma omp for schedule(static)
        for (int64_t y = 0; y < H; y++) {
            if (inu8) {
                const uint8_t *src = inu8 + y * W * 3;
                for (int64_t x = 0; x < W; x++) {
                    row4[x * LANES + 0] = lut[src[x * 3 + 0]];
                    row4[x * LANES + 1] = lut[src[x * 3 + 1]];
                    row4[x * LANES + 2] = lut[src[x * 3 + 2]];
                    row4[x * LANES + 3] = 0.0f;
                }
            } else {
                for (int c = 0; c < 3; c++) {
                    const float *src = inf + (c * H + y) * W;
                    for (int64_t x = 0; x < W; x++) row4[x * LANES + c] = src[x];
                }
                for (int64_t x = 0; x < W; x++) row4[x * LANES + 3] = 0.0f;
            }
            wpass_row4(row4, tmp + y * OW * LANES, OW, &tw);
        }
        free(row4);
    }
    hpass_interleaved(tmp, out, H, OH, OW, &th);
    free_taps(&tw); free_taps(&th);
    free(tmp);
    return 0;
}

int resize_aa_f32(const float *in, int64_t C, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out) {
    if (C != 3) return -2;
    return resize_interleaved(in, NULL, H, W, OH, OW, out);
}

int resize_aa_u8(const uint8_t *in, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out) {
    return resize_interleaved(NULL, in, H, W, OH, OW, out);
}
#endif /* ABOT_RESIZE_FAST */

#ifndef ABOT_RESIZE_FAST
/* in: [C,H,W] float32 contiguous -> out: [C,OH,OW]. Width pass first, then height (like aten). */
int resize_aa_f32(const float *in, int64_t C, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out) {
    float *tmp = (float *)malloc(sizeof(float) * C * H * OW);
    if (!tmp) return -1;
    taps_t tw = make_taps(W, OW), th = make_taps(H, OH);
    const int64_t L = C * H, nb = (L + RB - 1) / RB;
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; b++) {
        int64_t l0 = b * RB, nr = (L - l0 < RB) ? L - l0 : RB;
        wpass_rows(in + l0 * W, W, nr, tmp + l0 * OW, OW, &tw);
    }
    for (int64_t c = 0; c < C; c++)
        hpass_plane(tmp + c * H * OW, out + c * OH * OW, H, OH, OW, &th);
    free_taps(&tw); free_taps(&th);
    free(tmp);
    return 0;
}

/* in: [H,W,3] uint8 (PIL RGB) -> out: [3,OH,OW] float32, == resize(to_tensor(img)).
 * Rows are first converted to float (v/255.0f, exactly torch's to_tensor) into a per-thread
 * [RB, W] buffer, then handled by the same wpass_rows as the float path. */
int resize_aa_u8(const uint8_t *in, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out) {
    const int64_t C = 3;
    float *tmp = (float *)malloc(sizeof(float) * C * H * OW);
    if (!tmp) return -1;
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = (float)i / 255.0f;
    taps_t tw = make_taps(W, OW), th = make_taps(H, OH);
    const int64_t L = C * H, nb = (L + RB - 1) / RB;
#pragma omp parallel
    {
        float *rows = (float *)malloc(sizeof(float) * RB * W);
#pragma omp for schedule(static)
        for (int64_t b = 0; b < nb; b++) {
            int64_t l0 = b * RB, nr = (L - l0 < RB) ? L - l0 : RB;
            for (int64_t r = 0; r < nr; r++) {
                int64_t l = l0 + r, c = l / H, y = l % H;
                const uint8_t *src = in + (y * W) * C + c;
                float *row = rows + r * W;
                for (int64_t x = 0; x < W; x++) row[x] = lut[src[x * C]];
            }
            wpass_rows(rows, W, nr, tmp + l0 * OW, OW, &tw);
        }
        free(rows);
    }
    for (int64_t c = 0; c < C; c++)
        hpass_plane(tmp + c * H * OW, out + c * OH * OW, H, OH, OW, &th);
    free_taps(&tw); free_taps(&th);
    free(tmp);
    return 0;
}
#endif /* !ABOT_RESIZE_FAST */
