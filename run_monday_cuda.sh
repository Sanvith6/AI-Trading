#!/usr/bin/env bash
# ==============================================================================
# Monday US Trading Production Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if [ -f ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    elif [ -f "/home/sanvith-cf/laya/.venv/bin/python" ]; then
        PYTHON_BIN="/home/sanvith-cf/laya/.venv/bin/python"
    else
        echo "[ERROR] Python binary not found. Set PYTHON_BIN or activate a virtualenv."
        exit 1
    fi
fi

export PYTHONPATH=".:${PYTHONPATH}"
exec "${PYTHON_BIN}" -m trading.cli "$@"
