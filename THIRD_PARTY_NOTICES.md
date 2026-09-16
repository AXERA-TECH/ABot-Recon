# Third-Party Notices

ABot-Recon contains or depends on components with their own licenses. A notice
or license stated in an individual source file takes precedence for that file.

## Pi3

Core image encoding and geometric reconstruction code is derived from Pi3.

- Project: https://github.com/yyfz/Pi3
- Code license: BSD 3-Clause
- Released model weights: CC BY-NC 4.0
- Copyright: the Pi3 authors

## cuRoPE2D / CroCo / DUSt3R

CUDA cuRoPE2D sources under
“abot_recon/modeling/pi3/models/curope/” retain their upstream notices and
are licensed under CC BY-NC-SA 4.0.

- License: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- Copyright: NAVER Corporation and the respective upstream contributors

## DINOv2 and SALAD

The optional loop-closure backend contains an independently organized
SALAD-compatible descriptor network and loads external DINOv2 and SALAD
checkpoints. It does not vendor either upstream repository.

- DINOv2 project: https://github.com/facebookresearch/dinov2
- DINOv2 code license: Apache 2.0
- SALAD project: https://github.com/serizba/salad
- External checkpoints remain subject to the terms published by their
  respective authors and distributors.

## Sparse GPU Pose-Graph Optimization

The matrix-free block-sparse PCG implementation under
“abot_recon/sparse_loop/” is project code developed for ABot-Recon. It is
distributed under this repository's Apache License 2.0 and does not vendor
the HorizonStream loop implementation.

## FlashInfer

FlashInfer is an optional external dependency used by the paged-KV backend. It
is not vendored in this repository.

- Project: https://github.com/flashinfer-ai/flashinfer
- License: Apache 2.0

Evaluation-only repositories on the “eval” branch are external inputs. Their
own source and checkpoint licenses apply independently.
