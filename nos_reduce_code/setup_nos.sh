#!/usr/bin/env bash
set -euo pipefail

# Reuse the shared environment and install the NOS repo plus missing deps.
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
mkdir -p "$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating venv at $ENV_DIR"
  python3 -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e . \
  scikit-learn huggingface_hub hydra-core pytorch-lightning \
  einops h5py tqdm matplotlib

"$PY" - <<'PYEOF'
import torch
print("python:", __import__("sys").version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
import einops, h5py, sklearn, huggingface_hub, lightning, hydra, matplotlib
print("environment check: PASS")
PYEOF

echo "Environment ready: $ENV_DIR"
