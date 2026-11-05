#!/usr/bin/env bash
# Regenerate canonical backgrounds. Replaces matching files in the output directory.
# All arguments are forwarded to generate_canonical_h5.py; see --help.
set -euo pipefail

background_script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
background_repo_dir="$(CDPATH= cd -- "${background_script_dir}/../.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    background_python="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    background_python="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${background_repo_dir}/.venv/bin/python" ]]; then
    background_python="${background_repo_dir}/.venv/bin/python"
else
    background_python="python3"
fi

exec "${background_python}" "${background_script_dir}/generate_canonical_h5.py" --overwrite "$@"
