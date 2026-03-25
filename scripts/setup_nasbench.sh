#!/usr/bin/env bash
set -euo pipefail

REPO="/home/landos/Documents/Supplementary-material"
VENV="$REPO/.venv-nasbench"

if ! command -v paru >/dev/null 2>&1; then
  echo "[ERR] paru not found. Install paru first." >&2
  exit 1
fi

# 1) Install python311 via AUR
paru -S --needed --noconfirm python311 || { echo "[ERR] paru failed"; exit 1; }

# 2) Create venv
/usr/bin/python3.11 -m venv "$VENV"
source "$VENV/bin/activate"

# 3) TF + protobuf + project deps
pip install --upgrade pip
pip install tensorflow==2.15.0 protobuf==3.20.3
pip install -r "$REPO/source_code/requirement.txt"

# 4) Clone & install BB-DOB
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
pip install -e "$REPO/BB-DOB"

# 5) Clone & install nasbench
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
pip install -e "$REPO/nasbench"

# 6) Patch nasbench TF2 compat
python - <<'PY'
from pathlib import Path
import textwrap
init_path = Path("/home/landos/Documents/Supplementary-material/nasbench/nasbench/__init__.py")
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

# 7) Patch BB-DOB to accept absolute paths
python - <<'PY'
from pathlib import Path
path = Path("/home/landos/Documents/Supplementary-material/BB-DOB/bbdob/nas_bench_101.py")
text = path.read_text()
if "dataset_path" not in text:
    text = text.replace(
        "data_dir = \"{}/data\".format(os.path.dirname(os.path.abspath(__file__)))\n        if not os.path.exists(\"{}/{}\".format(data_dir, filename)):\n            os.makedirs(data_dir, exist_ok=True)\n            print(\"downloading data now...\")\n            subprocess.run(\"wget -P {} https://storage.googleapis.com/nasbench/{}\".format(data_dir, filename), shell=True)\n        self.nasbench = api.NASBench('{}/{}'.format(data_dir, filename))",
        "data_dir = \"{}/data\".format(os.path.dirname(os.path.abspath(__file__)))\n        dataset_path = filename\n        if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):\n            dataset_path = \"{}/{}\".format(data_dir, filename)\n        if not os.path.exists(dataset_path):\n            os.makedirs(data_dir, exist_ok=True)\n            print(\"downloading data now...\")\n            filename_only = os.path.basename(filename)\n            subprocess.run(\"wget -P {} https://storage.googleapis.com/nasbench/{}\".format(data_dir, filename_only), shell=True)\n            dataset_path = \"{}/{}\".format(data_dir, filename_only)\n        self.nasbench = api.NASBench(dataset_path)"
    )
    path.write_text(text)
PY

# 8) Download dataset
DATA_DIR="$REPO/source_code/instances/nasbench"
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/nasbench_full.tfrecord" ]; then
  curl -L -o "$DATA_DIR/nasbench_full.tfrecord" https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
fi

# 9) Quick smoke test
python "$REPO/source_code/main.py" problem=nasbench nb_instances_test=1 budget=50000

echo "[DONE] NASBench env setup complete."
