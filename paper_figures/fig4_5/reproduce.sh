#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${script_dir}/output}"

if command -v uv >/dev/null 2>&1; then
  uv run --script "${script_dir}/render.py" --output-dir "${output_dir}"
else
  python_bin="${PYTHON:-python3}"
  venv_dir="${script_dir}/.venv"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
    "${venv_dir}/bin/python" -m pip install --disable-pip-version-check \
      matplotlib==3.11.1 numpy==2.5.2
  fi
  "${venv_dir}/bin/python" "${script_dir}/render.py" --output-dir "${output_dir}"
fi
