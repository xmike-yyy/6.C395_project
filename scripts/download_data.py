"""
Extracts MIND-small zip files into data/train/MINDsmall_train/ and data/dev/MINDsmall_dev/.

Microsoft has restricted direct downloads. Download the zips manually first:
  1. Go to https://msnews.github.io/
  2. Download MINDsmall_train.zip and MINDsmall_dev.zip
  3. Place both files in the data/ directory
  4. Run this script

Safe to re-run — skips splits that already exist.
"""

import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# zip name → (extract-into dir, expected nested folder after extraction)
SPLITS = {
    "MINDsmall_train.zip": (DATA_DIR / "train", "MINDsmall_train"),
    "MINDsmall_dev.zip":   (DATA_DIR / "dev",   "MINDsmall_dev"),
}


def extract_split(zip_name: str, target_dir: Path, nested: str):
    final_dir = target_dir / nested
    zip_path = DATA_DIR / zip_name

    if final_dir.exists() and any(final_dir.iterdir()):
        print(f"[{nested}] already extracted — skipping")
        return

    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found.")
        print(f"       Download {zip_name} from https://msnews.github.io/ and place it in data/")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{nested}] extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    print(f"[{nested}] done → {final_dir}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    missing = [name for name in SPLITS if not (DATA_DIR / name).exists()]
    if missing:
        print("Missing zip files:", ", ".join(missing))
        print("Download them from https://msnews.github.io/ and place in the data/ directory.\n")

    for zip_name, (target_dir, nested) in SPLITS.items():
        extract_split(zip_name, target_dir, nested)

    all_done = all((target_dir / nested).exists() for _, (target_dir, nested) in SPLITS.items())
    if all_done:
        print("\nAll splits ready.")
    else:
        sys.exit(1)
