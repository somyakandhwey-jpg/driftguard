"""
data/download_elliptic.py

One-time helper to fetch the real Elliptic dataset from Kaggle into
data/raw/elliptic/. Everything else in this repo works on synthetic data
until you run this, so you can build/test the full pipeline first and only
do this when you're ready for real numbers.

Setup (one time):
  1. Create a Kaggle account, go to Account -> Create New API Token.
     This downloads kaggle.json.
  2. On Colab:
       from google.colab import files
       files.upload()   # upload kaggle.json
       !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     Locally:
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
  3. pip install kaggle
  4. Run this script: python data/download_elliptic.py
"""

import os
import subprocess
import zipfile

KAGGLE_DATASET = "ellipticco/elliptic-data-set"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "elliptic")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "elliptic-data-set.zip")

    print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", RAW_DIR],
        check=True,
    )

    if os.path.exists(zip_path):
        print("Unzipping...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        os.remove(zip_path)

    # Kaggle sometimes nests files in an extra folder - flatten if needed
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".csv") and root != RAW_DIR:
                os.rename(os.path.join(root, f), os.path.join(RAW_DIR, f))

    print("Done. Expect these files in", RAW_DIR)
    print(" - elliptic_txs_features.csv")
    print(" - elliptic_txs_classes.csv")
    print(" - elliptic_txs_edgelist.csv")


if __name__ == "__main__":
    main()
