# Project 1: NumPy histogram Newton boosting

Training and inference use **NumPy and the Python standard library only**. This is an original implementation of regularized second-order gradient-boosted trees, not the XGBoost package or a translation of its fitted model. All trees are newly trained from the official labeled data. The earlier external-library experiment is separate and is not part of this package.

## Authors

- Guodong MA — guodong.ma@epfl.ch
- Seohyun LEE — seo.lee@epfl.ch
- Anthony Yaghi — anthony.yaghi@epfl.ch

## Run

Use Python 3.12 and install `pip install -r requirements.txt`. Obtain the [official course data](https://raw.githubusercontent.com/epfml/ML_course/main/projects/project1/data/dataset_to_release.zip), extract it and put the three CSV files in `data/`.

Archive SHA256: `45812b9333df561c72bed861ba5ec382ddc2b68defd64ddda64118074d85a0fd`.

```bash
python run.py --data-dir data --output reproduced.csv
cmp submission.csv reproduced.csv
python run_baseline.py --data-dir data --output baseline_reproduced.csv
cmp submission_baseline.csv baseline_reproduced.csv
python optimize.py --data-dir data --output-dir optimization_rerun
python train.py --data-dir data --output-dir baseline_rerun
python -m unittest discover -s tests -v
```

Run these commands from the project directory. `run.py` verifies hashes and feature order and reproduces the tuning-selected ensemble in `submission.csv`. `run_baseline.py` reproduces the original model in `submission_baseline.csv`. `optimize.py` repeats the additional search and full-data training; `train.py` repeats the original baseline experiments. No pretrained XGBoost model, external ML code, external dataset or compiled ML dependency is used. NumPy's numerical backend is used through NumPy normally.

## Latest experiment: no established improvement

Eight additional configurations and two probability ensembles were evaluated. The tuning-selected system equally averages a 200-tree depth-four model (learning rate 0.1) and a 350-tree depth-three model (learning rate 0.05), both using 64 bins. Its cutoff is 0.20752688524570728. All settings and the full comparison table are in `optimization/results/metrics.json` and `optimization/results/comparison.csv`.

| Model | Tuning F1 | Internal holdout F1 | Holdout accuracy |
|---|---:|---:|---:|
| Original reference | 0.448261 | 0.444583 | 0.877901 |
| Tuning-selected ensemble | 0.449384 | 0.444475 | 0.876743 |

The holdout F1 difference is -0.000108, with paired-bootstrap 95% interval [-0.003443, +0.003363]. This run does **not** establish a gain. The ensemble remains the recorded tuning-selected experimental candidate; it is not claimed to be a better submission. The earlier model and predictions are preserved. Holdout scores were not used to switch to the retrospectively best variant. The team reported AIcrowd submission #338129 with F1 0.452 and accuracy 0.878 (three displayed decimals). This user-reported result is associated with the delivered ensemble CSV; it has not been independently verified on the platform. The original reference has no reported online score, so this does not establish an online improvement over that model. Details are in `optimization/results/online_result.json`.

See `optimization/README.md` for feature views, selection protocol, exact experiment source snapshots and cache integrity. `optimization/models/` holds the new models; `models/` holds the original ones. `optimization/results/` is the latest experiment; `results/` preserves baseline evidence. `report/` contains the updated two-page report; `report_baseline/` preserves the earlier report.

## Team foundation module and function entry points

`implementations.py` is now based on the teammate-supplied file. The exact original bytes are retained in `provenance/teammate_implementations.py.txt`. Integration changes to that module are formatting and a pseudoinverse fallback when the normal matrix is singular. Its six course function signatures and original helpers remain available for direct import.

Use the unified entry points for application code:

```python
from base_models import fit_model, predict_scores, METHODS

# tx_train/tx_valid must already be imputed and scaled using training statistics.
# Include a leading column of ones yourself if you want an intercept.
w, loss = fit_model("ridge_regression", y_train, tx_train, lambda_=1e-4)
scores = predict_scores("ridge_regression", tx_valid, w)

# Logistic targets must be 0/1. Scores below are probabilities.
w, loss = fit_model("reg_logistic_regression", y01_train, tx_train,
                    lambda_=1e-3, max_iters=1000, gamma=0.01)
probabilities = predict_scores("reg_logistic_regression", tx_valid, w)
```

`METHODS` contains `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, and `reg_logistic_regression`. `fit_model` returns `(w, loss)`; regularized losses exclude their penalties. Iterative methods accept `initial_w`, `max_iters` and `gamma`; regularized methods use `lambda_`. Optional `seed` controls SGD and restores the caller's NumPy random state. Missing values and invalid logistic labels are rejected. `predict_scores` does not choose a threshold; select it on tuning data using `pipeline.best_threshold`.

`train.py` uses these entry points for the ridge baseline. The additional boosting model remains in `numpy_boost.py` with its own unchanged numerical link function, so existing fitted-model predictions remain reproducible. Direct course calls such as `from implementations import ridge_regression` also work. The saved boosting models were trained before this integration; baseline equivalence and inference reproduction are checked in `results/team-integration.json`.

## Original baseline algorithm

`numpy_boost.py` constructs quantile bins using training features only (32 bins; up to 31 cuts). Low-cardinality numeric columns use midpoints between distinct values. Missing values receive bin zero. Each split tests both missing-left and missing-right, including missing-versus-observed splits. The gain is one half of `GL²/(HL+lambda) + GR²/(HR+lambda) - G²/(H+lambda)`; each leaf outputs `-G/(H+lambda)`. Logistic-loss gradient and Hessian are weighted for positive-class weighting. This is a simplified implementation, not feature-equivalent to the XGBoost library.

All configurations use learning rate 0.1, L2 coefficient 10, minimum child Hessian sum 20, row sampling 0.7, feature sampling 0.35 per tree, and seed 2026. Three candidates are predefined: depth 2, depth 3, and depth 3 with positive weight 3. Maximum rounds: 300. Every 10 rounds, tuning F1 is optimized over all distinct score thresholds; early stopping occurs after 60 rounds without a tuning F1 improvement above 1e-6. Only the best prefix of trees is retained. All unspecified parameters remain those of the constructor, recorded in saved JSON models.

The deterministic stratified split uses 60% train, 20% tune, 20% holdout. Quantile cuts and ridge imputation/scaling are fitted on train only. Candidate selection, stopping and thresholds use tune only. `results/selection.json` is written before boosting holdout evaluation. Final trees and bin edges are refitted from all labeled data for the selected round count; the selected cutoff is unchanged.

The holdout is the same split used in the earlier XGBoost experiment for comparability. Thus this is an internal development benchmark, not an untouched external test. No NumPy boosting configuration is selected or changed using its holdout score. The subsequent AIcrowd score is recorded separately from this local holdout; the evaluation subsets differ.

Ridge regression searches lambda in {0.0001, 0.01, 1}, with mean imputation, standardization and intercept. Entirely missing columns are imputed to zero. Survey codes remain numeric, and survey identifier features other than `Id` are retained. There is no geographic/temporal validation or external codebook preprocessing. The experiment classifies reported disease history, not future disease risk.

## Course checklist

- `implementations.py`: all six required NumPy methods, exact signatures, one-dimensional weights, batch-size-one SGD, MSE factor 1/2, and unpenalized returned losses. No `numpy.linalg.lstsq`.
- `run.py`: reproducible CSV, original test IDs/order, predictions {-1,+1}.
- `train.py`, `optimize.py`, `feature_views.py`, `pipeline.py`, `numpy_boost.py`: training and evaluation source.
- `optimization/models/`, `optimization/results/`: current weights, parameters, validation probabilities, measurements, hashes and verification evidence. `models/` and `results/` retain the baseline and split IDs.
- `report/main.tex`, `report/main.pdf`: English LaTeX report, at most two pages.

The library restriction is satisfied by this package. Page 3 of the supplied description permits any machine learning technique, while the six specified methods remain mandatory. Trees and neural networks implemented from scratch in NumPy are therefore permitted; external code and external datasets are not. Repository: [https://github.com/inori0cc/epfl-ml-project1-2026](https://github.com/inori0cc/epfl-ml-project1-2026). For the course form, use a URL pinned to the tested commit (`/tree/<full-commit-sha>`). **Course-system submission is still pending.** The AIcrowd username is `Guodong`; select submission **338129** as the final AIcrowd submission. All three author names and official email addresses are now recorded. AIcrowd submission #338129 was reported by the team at F1 0.452 and accuracy 0.878; older verification files retain their original pre-submission status. The supplied description states 29 October 2026 at 16:00 as the deadline. This implementation and report were developed with AI assistance, disclosed for the team to check against any additional course policy.

The official grading tests require development-only pytest, GitPython and Black; they are not runtime imports. Tests requiring a public GitHub link cannot pass before publication. All measurements in the report come from this NumPy run, not the earlier XGBoost run.

## References

- [Official project materials](https://github.com/epfml/ML_course/tree/main/projects/project1) and supplied Fall 2026 description.
- [Boosted-tree objective and derivation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html). Used for mathematical principles only; no third-party tree implementation copied.
- [Competition](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).
