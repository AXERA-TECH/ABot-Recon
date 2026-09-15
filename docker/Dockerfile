# ABot-Recon on Axera NPU — service image.
#
# The image carries Python, the dependencies, pyaxengine and this repo. The Axera runtime is NOT
# inside: it comes from the host at run time (the host has the driver installed):
#   AXCL card : -v /usr/lib/axcl:/usr/lib/axcl:ro -v /usr/bin/axcl:/usr/bin/axcl:ro
#               --device /dev/axcl_host --device /dev/ax_mmb_dev
#   AX650 chip: -v /soc:/soc:ro --privileged
# Models come from https://huggingface.co/AXERA-TECH/ABot-Recon, mounted at /models.
#
# build args: TARGET=axcl|ax650 (sets ABOT_DEVICE); PYAXENGINE_URL / PYAXVIDEO_BASE (wheel locations);
#             APT_MIRROR / PIP_INDEX_URL (optional mirrors for slow networks, e.g. mirrors.tuna.tsinghua.edu.cn)
ARG PY=3.12
FROM python:${PY}-slim-trixie AS build
ARG TARGETARCH
ARG PYAXENGINE_URL=https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc3/axengine-0.1.3-py3-none-any.whl
ARG PYAXVIDEO_BASE=https://github.com/AXERA-TECH/ax-video-sdk/releases/download/pyaxvideo-latest
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && cd /tmp && curl -fsSLO --retry 5 --retry-delay 5 --retry-all-errors "$PYAXENGINE_URL" && pip install --no-cache-dir /tmp/axengine-*.whl \
    && ARCH=$([ "$TARGETARCH" = "arm64" ] && echo aarch64 || echo x86_64) \
    && curl -fsSLO --retry 5 --retry-delay 5 --retry-all-errors "$PYAXVIDEO_BASE/pyaxvideo-0.1.1-py3-none-linux_${ARCH}.whl" \
    && pip install --no-cache-dir --no-deps /tmp/pyaxvideo-*.whl && cd /app
COPY abot_axera ./abot_axera
# compile the resize kernel now so the runtime image needs no compiler
RUN python -c "import abot_axera.preprocess as p; assert p._load_c() is not None, 'kernel build failed'" \
    && ls abot_axera/_build

FROM python:${PY}-slim-trixie
ARG PY
ARG TARGET=axcl
ARG APT_MIRROR=""
# libgomp: OpenMP runtime for the resize kernel
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/python${PY}/site-packages /usr/local/lib/python${PY}/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
WORKDIR /app
COPY abot_axera ./abot_axera
COPY --from=build /app/abot_axera/_build ./abot_axera/_build
COPY service ./service
COPY scripts ./scripts
COPY start_service.sh requirements.txt README.md ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh start_service.sh scripts/*.sh && mkdir -p /models /data/jobs
ENV ABOT_DEVICE=${TARGET} \
    ABOT_DEVICE_ID=0 \
    ABOT_MODELS=/models \
    ABOT_POSE_WEIGHTS=/models/host_pose_head/pose_head.safetensors \
    ABOT_POSE_CONFIG=/models/host_pose_head/pose_head_config.json \
    MAP_DATA=/data/jobs \
    MAP_PORT=8011 \
    VISER_PORT=8082 \
    PYBIN=/usr/local/bin \
    PYTHONUNBUFFERED=1
EXPOSE 8011 8082
VOLUME ["/data/jobs"]
ENTRYPOINT ["/entrypoint.sh"]
