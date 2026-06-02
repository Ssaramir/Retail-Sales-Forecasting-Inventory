# Data

The raw data is **not committed** to this repository because the Rossmann Store
Sales dataset is distributed under Kaggle competition rules.

## What you need

Place these two files in this folder:

- `train.csv` — daily sales per store (contains the `Sales` target)
- `store.csv` — one row per store with store-level metadata

`test.csv` is optional and not required (it has no `Sales` target, so it can't
be used for validation in this project).

## How to get them

### Option A — Kaggle API
1. Create an API token at kaggle.com → Settings → "Create New Token" (downloads `kaggle.json`).
2. Put `kaggle.json` in `~/.kaggle/` (Mac/Linux, then `chmod 600 ~/.kaggle/kaggle.json`)
   or `C:\Users\<you>\.kaggle\` (Windows).
3. Accept the rules at https://www.kaggle.com/competitions/rossmann-store-sales
4. Run:
   ```bash
   kaggle competitions download -c rossmann-store-sales -p data/
   unzip data/rossmann-store-sales.zip -d data/
   ```

### Option B — Manual
Download `train.csv` and `store.csv` from
https://www.kaggle.com/competitions/rossmann-store-sales/data and drop them in
this folder.

## Generated files
Running `python src/data_prep.py` creates `clean_merged.parquet` here. It is
gitignored because it is regenerated from the raw CSVs.
