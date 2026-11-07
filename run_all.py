"""run_all.py -- run the whole study:  python run_all.py --seeds 10

Results (CSV + PNG) are written to ./results.  If the UCI files are not found in
./data and cannot be downloaded, the synthetic fallback is used (and announced).
"""
import argparse
import os

import pandas as pd

import experiments as ex
import plots

pd.set_option("display.width", 200, "display.max_columns", 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-he", action="store_true", help="skip slow Paillier run")
    ap.add_argument("--only", nargs="*", default=["algorithms", "privacy", "security",
                                                  "communication", "framework"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seeds))
    raw, desc = ex.load_raw(a.data_dir, a.synthetic)
    print(f"\n=== DATA: {desc} ===")

    def save(name, df, hist=None, conv_names=None):
        df.to_csv(f"{a.out}/{name}.csv", index=False)
        print(f"\n--- {name} ---")
        print(ex.summarise(df))
        if hist and conv_names:
            plots.plot_convergence(hist, conv_names, "auc", f"{a.out}/{name}_auc.png",
                                   "global test AUC-ROC")

    if "algorithms" in a.only:
        df, h = ex.exp_algorithms(raw, seeds, a.rounds)
        save("exp1_algorithms", df, h, list(df.name.unique()))
        print(ex.significance(df, "FedAvg"))
    if "privacy" in a.only:
        df, h = ex.exp_privacy(raw, seeds, a.rounds, with_he=not a.skip_he)
        save("exp2_privacy", df)
        plots.plot_privacy_utility(df, f"{a.out}/exp2_privacy_utility.png")
    if "security" in a.only:
        df, h = ex.exp_security(raw, seeds, a.rounds)
        df.to_csv(f"{a.out}/exp3_security.csv", index=False)
        print("\n--- exp3_security (mean F1) ---")
        print(df.pivot_table(index="aggregator", columns="attack", values="f1").round(3))
        print("\n--- adversarial accuracy under FGSM ---")
        print(df[df.attack == "none"].groupby("aggregator")["adv_accuracy"].mean().round(3))
    if "communication" in a.only:
        df, h = ex.exp_communication(raw, seeds, a.rounds)
        save("exp4_communication", df, h, list(df.name.unique()))
    if "framework" in a.only:
        df, h = ex.exp_framework(raw, seeds, a.rounds)
        cols = ex.METRICS + ["conv_rounds", "comm_kb", "time_s", "epsilon",
                             "utility_loss_%", "poison_retention_%"]
        save("exp5_framework", df, h, list(df.name.unique()))
        print(df.groupby("name", sort=False)[cols].mean().round(3))
        print("\nSignificance vs FedAvg (F1):")
        print(ex.significance(df, "FedAvg", "f1"))
        print("\nSignificance vs FedProx (AUC):")
        print(ex.significance(df, "FedProx", "auc"))
        plots.plot_bars(df, "f1", f"{a.out}/exp5_f1.png", "F1 by configuration")


if __name__ == "__main__":
    main()
