"""Grid over weight decay (lambda) and hidden width (H) for the improved ANN.

Uses the same 10-fold x 5 repeat protocol and the same network recipe as
improvement/NF_RO_Improved.ipynb (4 LM restarts averaged, scaler fitted on the
training fold). Score = mean out-of-fold R over the three outputs.

The notebook uses lambda = 0.03, H = 8. Note that this grid is scored on the
same folds that the notebook reports, so the selected setting is optimistic.

Usage: python scripts/ann_grid.py
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["Time", "Pressure", "Temperature", "FeedFlow", "ConductivityFeed"]
OUTPUTS = ["PermeateConductivity", "PermeateFlowRate", "PermeateRecovery"]

LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
WIDTHS = [4, 6, 8, 10, 12]


def load_data():
    raw = pd.read_excel(ROOT / "improvement" / "Data_Desalination.xlsx", header=None)
    df = raw.iloc[5:, 3:].copy()
    df.columns = ["sn", "Time", "Pressure", "Temperature", "FeedFlow", "ConductivityFeed",
                  "PermeateRecovery", "PermeateFlowRate", "PermeateConductivity"]
    df = df.reset_index(drop=True).drop(columns="sn").astype(float)
    df.loc[np.isclose(df["Time"], 852.493), "Temperature"] = 29.2915
    return df[INPUTS].values, df[OUTPUTS].values


class Net:
    def __init__(self, H, seed):
        self.H = H
        r = np.random.default_rng(seed)
        # draw order matches _ANNCore in the notebook, so H = 8 gives identical weights
        self.t0 = np.concatenate([r.normal(scale=.3, size=5 * H), r.normal(scale=.3, size=H),
                                  r.normal(scale=.3, size=3 * H), r.normal(scale=.3, size=3)])

    def forward(self, X, t):
        H = self.H
        W1, b1 = t[:5 * H].reshape(H, 5), t[5 * H:6 * H]
        W2, b2 = t[6 * H:9 * H].reshape(3, H), t[9 * H:]
        return np.tanh(X @ W1.T + b1) @ W2.T + b2

    def fit(self, X, Y, lam, max_nfev=400):
        def resid(t):
            e = (self.forward(X, t) - Y).ravel()
            return e if lam == 0 else np.concatenate([e, np.sqrt(lam) * t])
        self.t = least_squares(resid, self.t0, method="lm", max_nfev=max_nfev).x
        return self


def fit_predict(X, Y, tr, te, lam, H, restarts=4, seed=1):
    xlo, xhi = X[tr].min(0), X[tr].max(0)
    ylo, yhi = Y[tr].min(0), Y[tr].max(0)
    xhi = np.where(xhi > xlo, xhi, xlo + 1)
    yhi = np.where(yhi > ylo, yhi, ylo + 1)
    Xtr = 2 * (X[tr] - xlo) / (xhi - xlo) - 1
    Ytr = 2 * (Y[tr] - ylo) / (yhi - ylo) - 1
    Xte = 2 * (X[te] - xlo) / (xhi - xlo) - 1
    nets = [Net(H, seed * 100 + s).fit(Xtr, Ytr, lam) for s in range(restarts)]
    P = np.mean([n.forward(Xte, n.t) for n in nets], axis=0)
    return (P + 1) * .5 * (yhi - ylo) + ylo


def score(X, Y, lam, H, n_splits=10, n_repeats=5, seed=42):
    splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                random_state=seed).split(X))
    preds = Parallel(n_jobs=-1)(delayed(fit_predict)(X, Y, tr, te, lam, H) for tr, te in splits)
    R = np.zeros((n_repeats, 3))
    for rep in range(n_repeats):
        oof = np.full_like(Y, np.nan)
        for k in range(n_splits):
            oof[splits[rep * n_splits + k][1]] = preds[rep * n_splits + k]
        R[rep] = [np.corrcoef(Y[:, j], oof[:, j])[0, 1] for j in range(3)]
    return R.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ann_grid.csv"))
    args = ap.parse_args()

    X, Y = load_data()
    rows = []
    for lam, H in itertools.product(LAMBDAS, WIDTHS):
        t = time.time()
        R = score(X, Y, lam, H)
        rows.append(dict(lam=lam, H=H, R_cond=R[0], R_flow=R[1], R_rec=R[2], R_mean=R.mean()))
        print(f"lambda={lam:<6g} H={H:<3d} mean R={R.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)

    res = pd.DataFrame(rows).sort_values("R_mean", ascending=False)
    print()
    print(res.pivot(index="lam", columns="H", values="R_mean").round(3).to_string())
    best = res.iloc[0]
    print(f"\nbest: lambda={best.lam:g}, H={int(best.H)}, mean R={best.R_mean:.3f}")

    Path(args.out).parent.mkdir(exist_ok=True)
    res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
