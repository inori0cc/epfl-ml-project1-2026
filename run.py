"""Reproduce the tuning-selected NumPy ensemble without retraining."""

from pathlib import Path
from run_optimized import main
from run_baseline import predict_model  # Preserve the single-model public helper.

if __name__ == "__main__":
    main(default_output=Path(__file__).resolve().parent / "submission.csv")
