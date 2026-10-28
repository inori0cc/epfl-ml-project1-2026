# EPFL Machine Learning Project 1

## Results

NumPy-only histogram gradient boosting ensemble.

| Evaluation | F1 | Accuracy |
|---|---:|---:|
| AIcrowd submission **338129** (Guodong) | **0.452** | **0.878** |
| Internal holdout | 0.444475 | 0.876743 |

Online scores were reported by the team. Methods and comparisons are in [the report](report/main.pdf), using the [official EPFL LaTeX template](https://github.com/epfml/ML_course/tree/main/projects/project1/latex-example-paper).

## Run

Use Python 3.12. Download the [official data](https://raw.githubusercontent.com/epfml/ML_course/main/projects/project1/data/dataset_to_release.zip) and extract `x_train.csv`, `y_train.csv`, and `x_test.csv` into `data/`.

From the repository root:

```bash
python -m pip install -r requirements.txt
python run.py --data-dir data --output reproduced.csv
cmp submission.csv reproduced.csv
```

To retrain the final two models with their frozen parameters, seed and threshold:

```bash
python train.py --data-dir data --output-dir retrained
python run.py --data-dir data --model-dir retrained --output retrained_predictions.csv
cmp submission.csv retrained_predictions.csv
```

`models/` contains the final ensemble only. `implementations.py` retains the six required course methods. Historical searches and tests remain available in earlier Git commits.
