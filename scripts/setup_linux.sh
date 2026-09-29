#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --disable-pip-version-check --progress-bar off -r requirements.txt
.venv/bin/python -m pip install --disable-pip-version-check --progress-bar off --no-deps -e .

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Next run: python -m teacher_reliability.env_check"
