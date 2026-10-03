"""Evaluate the MLR equations (9)-(11) printed in Adda et al. (2022) on their own Table 1.

Coefficients are copied from the paper. Inputs are used in the units of Table 1:
t time (h), P pressure (kg/cm2), T temperature (C), J_A feed flow (m3/h),
delta_A feed conductivity (uS/cm). load_data() applies the row 45 correction.

Usage: python scripts/mlr_equations.py
"""
import numpy as np

from split_search import load_data

# intercept, t, P, T, J_A, delta_A
EQUATIONS = {
    "PermeateConductivity": [-301.039, -0.53, 0.635, 0.536, 0.122, 0.070],   # Eq. (9)
    "PermeateRecovery": [111.49, 0.177, 0.195, -0.28, -0.36, -0.30],        # Eq. (10)
    "PermeateFlowRate": [2.68, 0.400, -0.0666, 0.1792, 0.205, 0.129],       # Eq. (11)
}
INPUTS = ["Time", "Pressure", "Temperature", "FeedFlow", "ConductivityFeed"]


def main():
    df = load_data()
    X = df[INPUTS].values
    for out, c in EQUATIONS.items():
        pred = c[0] + X @ np.array(c[1:])
        y = df[out].values
        R = np.corrcoef(y, pred)[0, 1]
        print(f"{out:22s} predicted [{pred.min():.0f}, {pred.max():.0f}]   "
              f"measured [{y.min():.2f}, {y.max():.2f}]   R = {R:.2f}")


if __name__ == "__main__":
    main()
