"""download_data.py -- fetch the UCI Heart Disease files and also save them as CSV.

The four files are plain text files on a public server, so no real "scraping"
is needed: we simply download them.   Usage:   python download_data.py
Creates   data/processed.<hospital>.data   (used by run_all.py)
and       data/<hospital>.csv              (same data, with column names).
"""
import os
import urllib.request

import pandas as pd

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]


def main(out="data"):
    os.makedirs(out, exist_ok=True)
    for h in HOSPITALS:
        name = f"processed.{h}.data"
        path = os.path.join(out, name)
        if not os.path.exists(path):
            print("downloading", BASE + name)
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
        df.to_csv(os.path.join(out, f"{h}.csv"), index=False)
        print(f"  {h}: {len(df)} patients -> {h}.csv")
    print("Done. Now run:  python run_all.py --seeds 10")


if __name__ == "__main__":
    main()
