#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
RECREATE=0

if [[ "${1:-}" == "--recreate" ]]; then
  RECREATE=1
fi

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      echo "Python launcher not found: ${PYTHON_BIN}" >&2
      exit 1
    fi
    command -v "${PYTHON_BIN}"
    return
  fi

  local active_venv="${VIRTUAL_ENV:-}"
  local candidates=()

  if command -v python3.13 >/dev/null 2>&1; then
    candidates+=("$(command -v python3.13)")
  fi
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    candidates+=("/opt/homebrew/bin/python3")
  fi
  if [[ -x /usr/local/bin/python3 ]]; then
    candidates+=("/usr/local/bin/python3")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  local candidate
  local version
  for candidate in "${candidates[@]}"; do
    if [[ -n "${active_venv}" && "${candidate}" == "${active_venv}"/* ]]; then
      continue
    fi
    version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${version}" == "3.13" ]]; then
      echo "${candidate}"
      return
    fi
  done

  for candidate in "${candidates[@]}"; do
    if [[ -n "${active_venv}" && "${candidate}" == "${active_venv}"/* ]]; then
      continue
    fi
    echo "${candidate}"
    return
  done

  echo "Could not find a usable python3 launcher." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
PYTHON_MAJOR_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "Using ${PYTHON_BIN} (${PYTHON_VERSION})"
if [[ "${PYTHON_MAJOR_MINOR}" != "3.13" ]]; then
  echo "Warning: Python 3.13 is the recommended local target for this repo. Continuing with ${PYTHON_MAJOR_MINOR}." >&2
fi

if [[ ${RECREATE} -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "Removing existing virtual environment at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Reusing existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install pytest python-chess h5py zstandard

python - <<'PY'
import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try:
    import torch

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
except Exception as exc:
    print(f"Torch import failed: {exc}")
PY
