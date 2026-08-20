#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${1:-${script_dir}/output/visual_examples.png}"

if command -v uv >/dev/null 2>&1; then
  uv run --script "${script_dir}/render.py" \
    --raw-dir "${script_dir}/data" \
    --output "${output}"
else
  python_bin="${PYTHON:-python3}"
  venv_dir="${script_dir}/.venv"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
    "${venv_dir}/bin/python" -m pip install --disable-pip-version-check \
      numpy==2.2.6 pillow==12.3.0
  fi
  "${venv_dir}/bin/python" "${script_dir}/render.py" \
    --raw-dir "${script_dir}/data" \
    --output "${output}"
fi

printf 'Fig. 6 written to %s\n' "${output}"
