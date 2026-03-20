#!/usr/bin/env bash
set -euo pipefail

REPO="${WORK:-$HOME}/Supplementary-material"
export REPO
ENV_NAME="ppo_env"
DATASET_DIR="${DATASET_DIR:-${WORK:-$HOME}/datasets}"
DATASET_FILE="${DATASET_FILE:-${DATASET_DIR}/nasbench_only108.tfrecord}"

if [ ! -d "$REPO" ]; then
  echo "[ERR] Repo not found at $REPO" >&2
  exit 1
fi

module purge
module load miniforge/24.9.0

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env create -f "$REPO/jeanzay/environment.yml"
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install tensorflow==2.15.0 protobuf==3.20.3
python -m pip install -r "$REPO/source_code/requirement.txt"

cd "$REPO"

if [ -d "$REPO/BB-DOB" ] && [ -z "$(ls -A "$REPO/BB-DOB")" ]; then
  rmdir "$REPO/BB-DOB"
fi
if [ ! -d "$REPO/BB-DOB" ]; then
  git clone https://github.com/e5120/BB-DOB
fi
if [ ! -f "$REPO/BB-DOB/setup.py" ] && [ ! -f "$REPO/BB-DOB/pyproject.toml" ]; then
  echo "[ERR] $REPO/BB-DOB missing setup.py/pyproject.toml. Remove the folder and re-run." >&2
  exit 1
fi
python -m pip install -e "$REPO/BB-DOB"

if [ -d "$REPO/nasbench" ] && [ -z "$(ls -A "$REPO/nasbench")" ]; then
  rmdir "$REPO/nasbench"
fi
if [ ! -d "$REPO/nasbench" ]; then
  git clone https://github.com/google-research/nasbench
fi
if [ ! -f "$REPO/nasbench/setup.py" ] && [ ! -f "$REPO/nasbench/pyproject.toml" ]; then
  echo "[ERR] $REPO/nasbench missing setup.py/pyproject.toml. Remove the folder and re-run." >&2
  exit 1
fi
python -m pip install -e "$REPO/nasbench"

python - <<'PY'
from pathlib import Path
import os
import textwrap

init_path = Path(os.environ["REPO"]) / "nasbench/nasbench/__init__.py"
text = init_path.read_text()
marker = "compatibility shim"
if marker not in text:
    patch = textwrap.dedent('''

"""Compatibility shim for running NASBench with TensorFlow 2.x."""

import sys

try:
    import tensorflow as _tf
    if not hasattr(_tf, "train") or not hasattr(_tf.train, "SessionRunHook"):
        import tensorflow.compat.v1 as _tf1
        _tf1.disable_v2_behavior()
        sys.modules["tensorflow"] = _tf1
except Exception:
    pass
''')
    init_path.write_text(text + patch)
PY

python - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["REPO"]) / "BB-DOB/bbdob/nas_bench_101.py"
text = path.read_text()
if "dataset_path" not in text:
    text = text.replace(
        "data_dir = \"{}/data\".format(os.path.dirname(os.path.abspath(__file__)))\n        if not os.path.exists(\"{}/{}\".format(data_dir, filename)):\n            os.makedirs(data_dir, exist_ok=True)\n            print(\"downloading data now...\")\n            subprocess.run(\"wget -P {} https://storage.googleapis.com/nasbench/{}\".format(data_dir, filename), shell=True)\n        self.nasbench = api.NASBench('{}/{}'.format(data_dir, filename))",
        "data_dir = \"{}/data\".format(os.path.dirname(os.path.abspath(__file__)))\n        dataset_path = filename\n        if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):\n            dataset_path = \"{}/{}\".format(data_dir, filename)\n        if not os.path.exists(dataset_path):\n            os.makedirs(data_dir, exist_ok=True)\n            print(\"downloading data now...\")\n            filename_only = os.path.basename(filename)\n            subprocess.run(\"wget -P {} https://storage.googleapis.com/nasbench/{}\".format(data_dir, filename_only), shell=True)\n            dataset_path = \"{}/{}\".format(data_dir, filename_only)\n        self.nasbench = api.NASBench(dataset_path)"
    )
    path.write_text(text)
PY

if [ ! -f "$DATASET_FILE" ]; then
  mkdir -p "$DATASET_DIR"
  wget -O "$DATASET_FILE" https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
fi

echo "[OK] NASBench env ready. Dataset: $DATASET_FILE"
