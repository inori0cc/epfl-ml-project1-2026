# Optimization experiment

This directory preserves all optimization trials separately from the original baseline. No online submissions or scores were used for model selection.

From the project root:

```bash
python run_optimized.py --data-dir data --output optimized_reproduced.csv
cmp optimization/submission.csv optimized_reproduced.csv
python optimize.py --data-dir data --output-dir optimization_rerun
```

The last command repeats the predefined search and full-data refits in a fresh directory. It uses the supplied original training-partition reference model to rank features. That model can be retrained with the original `train.py` workflow. A resumed search requires matching source/data/config fingerprints. Current code also verifies cached model hashes and recomputes tuning scores; older experiment records are deliberately not silently reused after source changes.

The exact source snapshots used for the delivered experiment are in `provenance/`, with hashes in `results/search_identity.json`. Subsequent source edits only add output-directory selection, cache integrity checks, recomputation of cached scalar scores, and formatting; they do not change the fresh-run training algorithm or search configurations.

Feature views are fitted on the training partition. The top-80 gain ranking uses the earlier training-fitted model; the previous choice of that reference involved tuning data. One-hot categories are learned from training features, without labels. The original numeric feature is retained alongside one-hot columns. A missing category gives NaN in the corresponding one-hot columns; an unseen category gives zeros. The additional missingness feature is the fraction of missing values among the selected 80 features.

All candidate models and probability arrays are retained. Equal-weight averages of the best two and three single models are also evaluated on tune. Model, tree count, ensemble membership and cutoff are fixed before new holdout scoring. Full-data refitting retains feature views but refits bin boundaries and trees.

The holdout was already inspected during earlier work, so these are internal development comparisons. Tuning F1 is optimistic after trying multiple configurations. `results/paired_comparison.json` reports paired bootstrap uncertainty, conditional on the fitted models. It does not correct for selection or holdout reuse. More complex models and feature engineering are not assumed to help; all unsuccessful variants remain in the comparison table.

`results/missingness_shift.csv` is a descriptive comparison of unlabeled train/test missingness. It did not affect any selected feature, model or threshold.
