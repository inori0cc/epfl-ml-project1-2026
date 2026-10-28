# EPFL Machine Learning Project 1

## Results

NumPy-only histogram gradient boosting ensemble.

| Evaluation | F1 | Accuracy |
|---|---:|---:|
| AIcrowd submission **338129** (Guodong) | **0.452** | **0.878** |
| Internal holdout | 0.444475 | 0.876743 |

The online scores were reported by the team. Full experiment results and methods are in [the report](report/main.pdf).

## Run

Use Python 3.12. Download the [official dataset](https://raw.githubusercontent.com/epfml/ML_course/main/projects/project1/data/dataset_to_release.zip) and extract `x_train.csv`, `y_train.csv`, and `x_test.csv` into `data/`.

From the repository root:

```bash
python -m pip install -r requirements.txt
python run.py --data-dir data --output reproduced.csv
cmp submission.csv reproduced.csv
```

`run.py` loads the saved models without retraining and reproduces the submitted predictions. Runtime dependencies are NumPy and the Python standard library.

To repeat training and experiments in fresh output directories:

```bash
python train.py --data-dir data --output-dir baseline_rerun
python optimize.py --data-dir data --output-dir optimization_rerun
```

To run the project tests:

```bash
python -m unittest discover -s tests -v
```
