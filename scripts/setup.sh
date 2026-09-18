#!/usr/bin/env bash
# Fetch the MyGOChat model into ./vendor and install Python dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${MYGOCHAT_PATH:-$REPO_ROOT/vendor/MyGOChat}"
MYGOCHAT_REMOTE="${MYGOCHAT_REMOTE:-https://github.com/qaz45647/MyGOChat.git}"
PYTHON="${PYTHON:-python3}"
SKIP_MODEL="${SKIP_MODEL:-0}"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required."

info "Installing bot dependencies"
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt"

if [[ "$SKIP_MODEL" == "1" ]]; then
  info "SKIP_MODEL=1 -- skipping torch and the checkpoint. Use MYGO_ENGINE=random."
  exit 0
fi

# The 1.3 GB model.safetensors is a git-lfs object. Without git-lfs the clone
# "succeeds" but leaves a ~130-byte pointer file, and transformers fails later
# with a confusing deserialization error.
if ! command -v git-lfs >/dev/null 2>&1; then
  die "git-lfs is required for the 1.3 GB checkpoint.
  Debian/Ubuntu: sudo apt-get install git-lfs
  macOS:         brew install git-lfs
  Then re-run this script."
fi
git lfs install --skip-repo

info "Installing model dependencies (this pulls torch; it is large)"
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements-model.txt"

if [[ -d "$VENDOR_DIR/.git" ]]; then
  info "Updating existing checkout at $VENDOR_DIR"
  git -C "$VENDOR_DIR" pull --ff-only
else
  info "Cloning MyGOChat into $VENDOR_DIR"
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone --depth 1 "$MYGOCHAT_REMOTE" "$VENDOR_DIR"
fi

info "Pulling LFS objects (~1.3 GB, this takes a while)"
git -C "$VENDOR_DIR" lfs pull

MODEL_FILE="$VENDOR_DIR/mygochat/models/model.safetensors"
[[ -f "$MODEL_FILE" ]] || die "Expected $MODEL_FILE to exist after the clone."

SIZE=$(wc -c < "$MODEL_FILE")
if (( SIZE < 1000000 )); then
  die "$MODEL_FILE is only $SIZE bytes -- that is still an LFS pointer.
  Run: git -C \"$VENDOR_DIR\" lfs pull"
fi

info "Model ready ($(( SIZE / 1024 / 1024 )) MB)."
info "Next: cp .env.example .env, add DISCORD_TOKEN, then run 'python -m bot'."
