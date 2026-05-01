#!/usr/bin/env sh
set -eu

if ! command -v python >/dev/null 2>&1; then
  echo "bootstrap.sh requires python on PATH." >&2
  exit 1
fi

python - <<'PY'
from src.runtime_assets import ensure_supported_bootstrap_platform
ensure_supported_bootstrap_platform()
PY

echo "Installing workshop requirements..."
python -m pip install -r requirements.txt

python - <<'PY'
from src.runtime_assets import (
    get_dataset_asset_descriptor,
    get_dataset_asset_url,
    get_tft_asset_descriptor,
    get_tft_asset_url,
)

dataset = get_dataset_asset_descriptor()
tft = get_tft_asset_descriptor()

print(f"Dataset release tag: {dataset.release_tag}")
print(f"Dataset asset filename: {dataset.filename}")
print(f"Dataset asset URL: {get_dataset_asset_url()}")
print(f"TFT release tag: {tft.release_tag}")
print(f"TFT bundle filename: {tft.filename}")
print(f"TFT asset URL: {get_tft_asset_url()}")
print(f"Dataset install path: {dataset.local_install_path}")
print(f"TFT install root: {tft.local_install_path}")
PY

echo "bootstrap.sh is wired to the canonical runtime contract."
echo "Use the shared src/ asset helpers during rehearsal or published setup."
