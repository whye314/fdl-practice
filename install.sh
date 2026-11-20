#!/usr/bin/env bash
# Run from a writable checkout with internet access.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv_bin="$PWD/.notebook-tools/bin/uv"
if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ ! -x "$uv_bin" ]]; then
    python3 -m pip install --target "$PWD/.notebook-tools" uv
fi

# Keep the repository's PyTorch index and pin. Use a Python version with
# wheels for PyTorch 2.6 even if the notebook's interpreter is newer.
UV_PROJECT_ENVIRONMENT="$PWD/.venv" "$uv_bin" sync --python 3.11 --no-install-project

"$PWD/.venv/bin/python" - <<'PYTHON'
import torch

print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
else:
    print("CUDA unavailable. Select a GPU accelerator before running training.")
PYTHON
