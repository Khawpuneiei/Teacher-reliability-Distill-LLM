#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The vLLM lock targets Linux x86_64 only." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'

if [[ ! -x .venv-vllm/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv-vllm
fi
VLLM_PYTHON="$REPO_ROOT/.venv-vllm/bin/python"

if [[ "${RESOLVE_VLLM_LOCK:-0}" == "1" || ! -f requirements-vllm.lock ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    "$VLLM_PYTHON" -m pip install --disable-pip-version-check uv
    UV_BIN="$REPO_ROOT/.venv-vllm/bin/uv"
  else
    UV_BIN="$(command -v uv)"
  fi
  "$UV_BIN" pip compile requirements-vllm.in \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    --generate-hashes \
    --output-file requirements-vllm.lock
fi

"$VLLM_PYTHON" -m pip install --disable-pip-version-check --require-hashes -r requirements-vllm.lock
"$VLLM_PYTHON" -m pip install --disable-pip-version-check --no-deps -e .
"$VLLM_PYTHON" -c 'import bitsandbytes, importlib.metadata, teacher_reliability.vllm_worker, vllm; assert importlib.metadata.version("vllm-bnb-plugin") == "0.0.3"'

echo "vLLM environment ready. Activate with: source .venv-vllm/bin/activate"
