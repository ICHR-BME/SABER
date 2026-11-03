#!/usr/bin/env bash
# One command on a rented Linux GPU box (e.g. RTX 4090): install, then run LIBERO-Goal S0/S1/S2
# in English + 4 languages with lerobot/smolvla_libero. Does nothing without a CUDA GPU.
#   bash scripts/gpu_libero.sh [INTERPRETER]   (default: hf:Qwen/Qwen2.5-1.5B-Instruct)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo "no NVIDIA GPU found; aborting"; exit 2; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export MUJOCO_GL=egl   # headless rendering
uv sync --extra sim --extra interp
INTERP="${1:-hf:Qwen/Qwen2.5-1.5B-Instruct}"
uv run python scripts/eval_libero.py --systems S0,S1,S2 --langs en,es,pt,fr,de --episodes 20 \
  --interpreter "$INTERP" --out outputs/libero
