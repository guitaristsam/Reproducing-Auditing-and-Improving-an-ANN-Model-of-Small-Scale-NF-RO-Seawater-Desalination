"""Search for the 60/13 split used in reproduction/1_Paper_Reproduction.ipynb.

Adda et al. do not say which 13 rows were held out, so we look for the
train_test_split random_state whose in-sample R values come closest to their
Table 2. Two stages:

  1. MLR only, random_state 0..7999, score = mean |dR| over the 3 MLR rows
  2. the 12-restart ANN from notebook 1 on the best 10 seeds from stage 1,
     score = mean |dR| over all 6 rows

Usage: python scripts/split_search.py [--n-seeds 8000] [--top 10]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["Time", "Pressure", "Temperature", "FeedFlow", "ConductivityFeed"]
OUTPUTS = ["PermeateConductivity", "PermeateFlowRate", "PermeateRecovery"]

# Table 2 of Adda et al. (2022)
PAPER_R = {("PermeateConductivity", "ANN"): 0.9691, ("PermeateConductivity", "MLR"): 0.6220,
           ("PermeateFlowRate", "ANN"): 0.9422, ("PermeateFlowRate", "MLR"): 0.6790,
           ("PermeateRecovery", "ANN"): 0.9636, ("PermeateRecovery", "MLR"): 0.5720}


def load_data():
    raw = pd.read_excel(ROOT / "reproduction" / "Data_Desalination.xlsx", header=None)
    df = raw.iloc[5:, 3:].copy()
    df.columns = ["sn", "Time", "Pressure", "Temperature", "FeedFlow", "ConductivityFeed",
                  "PermeateRecovery", "PermeateFlowRate", "PermeateConductivity"]
    df = df.reset_index(drop=True).drop(columns="sn").astype(float)
    df.loc[np.isclose(df["Time"], 852.493), "Temperature"] = 29.2915
    return df


def scale(d):
    return 2 * (d - d.min()) / (d.max() - d.min()) - 1


def forward(X, t):
    W1, b1 = t[:40].reshape(8, 5), t[40:48]
    W2, b2 = t[48:72].reshape(3, 8), t[72:75]
    return np.tanh(X @ W1.T + b1) @ W2.T + b2


def fit_ann(X, Y, n_restarts=12):
    # same initialisation and LM settings as the ANN class in notebook 1
    best, best_sse = None, np.inf
    for s in range(n_restarts):
        r = np.random.default_rng(s)
        t0 = np.concatenate([r.normal(scale=.5, size=40), r.normal(scale=.5, size=8),
                             r.normal(scale=.5, size=24), r.normal(scale=.5, size=3)])
        res = least_squares(lambda t: (forward(X, t) - Y).ravel(), t0, method="lm", max_nfev=3000)
        sse = np.sum((forward(X, res.x) - Y) ** 2)
        if sse < best_sse:
            best, best_sse = res.x, sse
    return best


def split(Xn, Yn, seed):
    X_tr, _, Y_tr, _ = train_test_split(Xn, Yn, test_size=13, random_state=seed, shuffle=True)
    return X_tr.values, Y_tr.values


def r_per_output(Y, P):
    # R is unchanged by the linear de-normalisation, so scaled values are fine here
    return [np.corrcoef(Y[:, j], P[:, j])[0, 1] for j in range(3)]


def mlr_score(Xn, Yn, seed):
    X, Y = split(Xn, Yn, seed)
    R = r_per_output(Y, LinearRegression().fit(X, Y).predict(X))
    return np.mean([abs(R[j] - PAPER_R[(o, "MLR")]) for j, o in enumerate(OUTPUTS)])


def full_score(Xn, Yn, seed):
    X, Y = split(Xn, Yn, seed)
    R_mlr = r_per_output(Y, LinearRegression().fit(X, Y).predict(X))
    R_ann = r_per_output(Y, forward(X, fit_ann(X, Y)))
    d = [abs(R_ann[j] - PAPER_R[(o, "ANN")]) for j, o in enumerate(OUTPUTS)]
    d += [abs(R_mlr[j] - PAPER_R[(o, "MLR")]) for j, o in enumerate(OUTPUTS)]
    return dict(seed=seed, mean_abs_dR=np.mean(d), max_abs_dR=np.max(d),
                R_ann_cond=R_ann[0], R_ann_flow=R_ann[1], R_ann_rec=R_ann[2],
                R_mlr_cond=R_mlr[0], R_mlr_flow=R_mlr[1], R_mlr_rec=R_mlr[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "split_search.csv"))
    args = ap.parse_args()

    df = load_data()
    Xn, Yn = scale(df[INPUTS]), scale(df[OUTPUTS])

    stage1 = pd.DataFrame({"seed": range(args.n_seeds),
                           "mlr_mean_abs_dR": [mlr_score(Xn, Yn, s) for s in range(args.n_seeds)]})
    top = stage1.nsmallest(args.top, "mlr_mean_abs_dR")
    print(f"stage 1: {args.n_seeds} seeds, best {args.top} by MLR |dR|:")
    print(top.to_string(index=False))

    rows = Parallel(n_jobs=-1)(delayed(full_score)(Xn, Yn, s) for s in top.seed)
    stage2 = pd.DataFrame(rows).merge(stage1, on="seed").sort_values("mean_abs_dR")
    print("\nstage 2: ANN + MLR, all six rows")
    print(stage2.round(4).to_string(index=False))

    # reference point: the seed used before the search
    ref = full_score(Xn, Yn, 42)
    print(f"\nseed 42 (no search): mean |dR| = {ref['mean_abs_dR']:.4f}, max = {ref['max_abs_dR']:.4f}")
    print(f"selected seed: {stage2.seed.iloc[0]}")

    Path(args.out).parent.mkdir(exist_ok=True)
    stage2.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
