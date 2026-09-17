# Stage 1: fetch MyGOChat including the 1.3 GB LFS checkpoint.
FROM python:3.11-slim AS model

RUN apt-get update \
 && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG MYGOCHAT_REMOTE=https://github.com/qaz45647/MyGOChat.git
RUN git lfs install --skip-repo \
 && git clone --depth 1 "$MYGOCHAT_REMOTE" /vendor/MyGOChat \
 && git -C /vendor/MyGOChat lfs pull \
 && rm -rf /vendor/MyGOChat/.git

# Stage 2: the runtime image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MYGOCHAT_PATH=/app/vendor/MyGOChat \
    SETTINGS_PATH=/app/data/guilds.json \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY requirements.txt requirements-model.txt ./
# torch comes from the CPU-only index (a fraction of the default CUDA build's
# size); everything else comes from PyPI. These must be two steps: --index-url
# REPLACES PyPI rather than adding to it, and transformers is not published on
# download.pytorch.org, so a single combined install cannot resolve it.
# Installing torch first also means the second command sees torch>=2.1 already
# satisfied and leaves the CPU wheel in place.
# For a GPU host, drop the first pip install and let the second pull CUDA torch.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements-model.txt

COPY --from=model /vendor /app/vendor
COPY bot/ ./bot/

RUN useradd --create-home --uid 10001 mygo \
 && mkdir -p /app/data \
 && chown -R mygo:mygo /app/data
USER mygo

VOLUME ["/app/data"]

CMD ["python", "-m", "bot"]
