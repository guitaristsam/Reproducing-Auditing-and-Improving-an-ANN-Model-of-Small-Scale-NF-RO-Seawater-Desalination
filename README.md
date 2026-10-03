# NF/RO desalination ANN: reproduction and re-evaluation

Code and data for the paper *Reproducing, Auditing, and Improving an ANN Model of Small-Scale
NF/RO Seawater Desalination* (S. Anand, P. Achanta, K. Nagelia, A. Karnik, B. Suthar,
R. Mahadeva).

The paper reproduces the 5-8-3 ANN of Adda et al. (2022), checks the published results, and then
scores the same model and several alternatives with repeated cross-validation.

## Contents

```
reproduction/1_Paper_Reproduction.ipynb   Notebook 1: reproduces Table 2 and Fig. 3 of Adda et al.
improvement/NF_RO_Improved.ipynb          Notebook 2: cross-validated comparison, intervals, sensitivity
scripts/split_search.py                   search for the 60/13 split used in Notebook 1
scripts/ann_grid.py                       weight decay / hidden width grid for the improved ANN
scripts/lm_cap.py                         effect of the LM evaluation cap on the paper's ANN recipe
scripts/mlr_equations.py                  evaluates the MLR equations (9)-(11) printed in the paper
scripts/make_methodology_figure.py        Figure 1 of the paper
```

## Setup

Tested with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

The data file is not included (see Data below). Put `Data_Desalination.xlsx` in both
`reproduction/` and `improvement/`, then run each notebook from its own folder:

```bash
cd reproduction
jupyter nbconvert --to notebook --execute --inplace 1_Paper_Reproduction.ipynb

cd ../improvement
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 NF_RO_Improved.ipynb
```

Notebook 1 takes under a minute, Notebook 2 a few minutes (cross-validation folds run in parallel
on all cores). Tables and figures are written next to each notebook.

The scripts are run from the repository root, read the same data file and write to `results/`:

```bash
python scripts/split_search.py      # ~2 min, selects random_state 3860
python scripts/ann_grid.py          # ~5 min, selects lambda = 0.03, H = 8
python scripts/lm_cap.py            # ~1.5 min
python scripts/mlr_equations.py     # prints predicted vs measured ranges
python scripts/make_methodology_figure.py results
```

Timings are from a 28-core machine.

## Data

The data are taken from Table 1 of

> A. Adda, S. Hanini, S. Bezari, M. Laidi, M. Abbas, Modeling and optimization of small-scale
> NF/RO seawater desalination using the artificial neural network (ANN),
> *Environmental Engineering Research* 27 (2022) 200383. https://doi.org/10.4491/eer.2020.383

The article is openly available at https://www.eeer.org/upload/eer-2020-383.pdf. The data file
is not redistributed here.

The code expects the 73 rows of Table 1 in `Data_Desalination.xlsx`, first sheet, rows 6-78,
columns D-L, in this order: s/n, time (h), pressure (kg/cm²), temperature (°C), feed flow (m³/h),
feed conductivity (µS/cm), permeate recovery (%), permeate flow rate (m³/h), permeate
conductivity (µS/cm). This is the column order of Table 1. Rows 1-5 and columns A-C are ignored.

Two corrections are applied in code: the `s/n` column is dropped, and the row 45 feed temperature
(3.2915 °C) is replaced with 29.2915 °C.

## License

The code is released under the MIT License (see `LICENSE`).
