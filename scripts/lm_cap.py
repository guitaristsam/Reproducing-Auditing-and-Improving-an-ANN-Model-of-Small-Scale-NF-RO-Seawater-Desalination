"""Sensitivity of the paper's ANN recipe (ablation step A0) to the LM evaluation cap.

A0: scaler fitted on all 73 rows, no weight decay, best of 4 restarts, scored with the
same 10-fold x 5 repeat protocol as improvement/NF_RO_Improved.ipynb. The notebook uses a
cap of 400.

Usage: python scripts/lm_cap.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold

from ann_grid import Net, load_data

ROOT = Path(__file__).resolve().parents[1]
CAPS = [100, 400, 3000]


def fit_predict(X, Y, tr, te, cap, restarts=4, seed=1):
    xlo, xhi, ylo, yhi = X.min(0), X.max(0), Y.min(0), Y.max(0)
    Xtr = 2 * (X[tr] - xlo) / (xhi - xlo) - 1
    Ytr = 2 * (Y[tr] - ylo) / (yhi - ylo) - 1
    Xte = 2 * (X[te] - xlo) / (xhi - xlo) - 1
    nets = [Net(8, seed * 100 + s).fit(Xtr, Ytr, 0.0, max_nfev=cap) for s in range(restarts)]
    best = min(nets, key=lambda n: np.sum((n.forward(Xtr, n.t) - Ytr) ** 2))
    P = best.forward(Xte, best.t)
    return (P + 1) * .5 * (yhi - ylo) + ylo


def score(X, Y, cap, n_splits=10, n_repeats=5, seed=42):
    splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                random_state=seed).split(X))
    preds = Parallel(n_jobs=-1)(delayed(fit_predict)(X, Y, tr, te, cap) for tr, te in splits)
    R = np.zeros((n_repeats, 3))
    for rep in range(n_repeats):
        oof = np.full_like(Y, np.nan)
        for k in range(n_splits):
            oof[splits[rep * n_splits + k][1]] = preds[rep * n_splits + k]
        R[rep] = [np.corrcoef(Y[:, j], oof[:, j])[0, 1] for j in range(3)]
    return R.mean(0)


def main():
    X, Y = load_data()
    rows = []
    for cap in CAPS:
        R = score(X, Y, cap)
        rows.append(dict(max_nfev=cap, R_cond=R[0], R_flow=R[1], R_rec=R[2], R_mean=R.mean()))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "lm_cap.csv", index=False)


if __name__ == "__main__":
    main()
