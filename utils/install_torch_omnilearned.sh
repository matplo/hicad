#!/usr/bin/env bash
# Installs PyTorch with GPU acceleration (CUDA on Linux, MPS on macOS) and omnilearned.
#
# On macOS: installs the official PyPI wheels, which already bundle Metal/MPS
# support on Apple Silicon - no extra index needed.
#
# On Linux: probes `nvidia-smi` for the max CUDA version the installed driver
# supports, cross-references that against the CUDA wheel tags currently
# published on download.pytorch.org, and installs the newest compatible one.
# Falls back to CPU-only wheels if no NVIDIA driver is found.
#
# Override the interpreter with PYTHON_BIN=... if `python3` isn't the right one.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_PACKAGES=(torch torchvision torchaudio)
PYTORCH_INDEX_ROOT="https://download.pytorch.org/whl"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "'$PYTHON_BIN' not found. Set PYTHON_BIN to your python executable."
"$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || die "pip is not available for $PYTHON_BIN."

PYVER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
  || warn "omnilearned requires Python >=3.10; detected $PYVER. Continuing anyway."

pip_install() {
  log "pip install $*"
  "$PYTHON_BIN" -m pip install --upgrade "$@"
}

install_macos() {
  local arch
  arch="$(uname -m)"
  log "macOS detected (arch: $arch)."
  pip_install "${TORCH_PACKAGES[@]}"

  if [[ "$arch" == "arm64" ]]; then
    "$PYTHON_BIN" - <<'PY'
import torch
print(f"torch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
PY
  else
    warn "Intel Mac detected: MPS is not available on this hardware, PyTorch will run on CPU."
  fi
}

detect_driver_cuda_version() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+' || echo ""
}

# Prints space-separated CUDA wheel tags (e.g. "cu128 cu126 cu121"), newest
# compatible first, by comparing against tags currently live on PyTorch's
# wheel index. Empty output means "couldn't determine, fall back to CPU".
pick_cuda_tags() {
  local driver_version="$1"
  local tmp
  tmp="$(mktemp)"
  if ! curl -fsSL "$PYTORCH_INDEX_ROOT/" -o "$tmp" 2>/dev/null; then
    rm -f "$tmp"
    echo ""
    return
  fi

  "$PYTHON_BIN" - "$tmp" "$driver_version" <<'PY'
import re, sys

path, driver = sys.argv[1], sys.argv[2]
html = open(path).read()
tags = sorted(set(re.findall(r'href="(cu\d+)/"', html)))

def tag_version(tag):
    digits = tag[2:]
    return (int(digits[:-1]), int(digits[-1]))

d_major, d_minor = (int(x) for x in driver.split('.')[:2])
compatible = [t for t in tags if tag_version(t) <= (d_major, d_minor)]
compatible.sort(key=tag_version, reverse=True)
# Only the newest few matter in practice; older tags are legacy CUDA versions
# with no wheels for current Python/arch combos, not worth cycling through.
print(' '.join(compatible[:3]))
PY
  rm -f "$tmp"
}

install_linux() {
  log "Linux detected."
  local driver_version
  driver_version="$(detect_driver_cuda_version)"

  if [[ -z "$driver_version" ]]; then
    warn "No NVIDIA driver detected (nvidia-smi missing or failed) - installing CPU-only PyTorch."
    pip_install "${TORCH_PACKAGES[@]}" --index-url "$PYTORCH_INDEX_ROOT/cpu"
    return
  fi

  log "NVIDIA driver reports max supported CUDA version: $driver_version"
  local candidates
  candidates="$(pick_cuda_tags "$driver_version")"

  if [[ -z "$candidates" ]]; then
    warn "Could not reach download.pytorch.org to pick a CUDA tag - falling back to CPU-only PyTorch."
    pip_install "${TORCH_PACKAGES[@]}" --index-url "$PYTORCH_INDEX_ROOT/cpu"
    return
  fi

  log "Compatible CUDA wheel tags (newest first): $candidates"
  local tag installed=0
  for tag in $candidates; do
    log "Attempting PyTorch install for $tag..."
    if pip_install "${TORCH_PACKAGES[@]}" --index-url "$PYTORCH_INDEX_ROOT/$tag"; then
      if "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        log "PyTorch with CUDA ($tag) installed and GPU is visible."
        installed=1
        break
      else
        warn "$tag installed but torch.cuda.is_available() is False - trying an older CUDA tag."
      fi
    else
      warn "pip install failed for $tag - trying an older CUDA tag."
    fi
  done

  if [[ "$installed" -ne 1 ]]; then
    warn "Could not get a working CUDA build; falling back to CPU-only PyTorch."
    pip_install "${TORCH_PACKAGES[@]}" --index-url "$PYTORCH_INDEX_ROOT/cpu"
  fi
}

main() {
  local os
  os="$(uname -s)"
  case "$os" in
    Darwin) install_macos ;;
    Linux)  install_linux ;;
    *) die "Unsupported OS: $os" ;;
  esac

  log "Installing omnilearned..."
  pip_install omnilearned

  log "Verifying installation..."
  "$PYTHON_BIN" - <<'PY'
import torch
print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU backend: CUDA ({torch.cuda.get_device_name(0)})")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("GPU backend: MPS (Apple Silicon)")
else:
    print("No GPU backend detected - torch will run on CPU")

try:
    import omnilearned  # noqa: F401
    print("omnilearned import OK")
except ImportError as exc:
    print(f"omnilearned installed but failed to import: {exc}")
PY

  log "Done."
}

main "$@"
