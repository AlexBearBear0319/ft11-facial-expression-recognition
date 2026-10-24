# Selected validation evidence

These files come from saved runs. We re-scored each saved checkpoint on its recorded validation rows and matched its accuracy, macro-F1, loss, and confusion matrix. No FER2013 source-test metric is included.

| Evidence | Run | Training data | Validation data | Budget |
| --- | --- | ---: | ---: | --- |
| `pilot_validation/` | `20261024-190809-840000` | 512 | 512 | 1 epoch |
| `full_budget_validation/` | `20261024-191418-838000` | 23,743 | 4,190 | Up to 10 epochs; early stopping |

## Saved validation scores

| Run | Model | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Pilot | Majority reference | 25.78% | 5.86% |
| Pilot | MLP | 25.78% | 5.86% |
| Pilot | Original small CNN | 17.58% | 4.27% |
| Pilot | ResNet18 | 25.78% | 5.86% |
| Full budget | Majority reference | 25.61% | 5.83% |
| Full budget | MLP | 36.37% | 27.00% |
| Full budget | Original small CNN | 47.33% | 37.38% |
| Full budget | ResNet18 | 53.99% | 49.71% |

Each run folder has a compact comparison, run summary, per-class metrics, confusion matrices, training history, and loss and validation-score plots. `shared/` has the split summary and class-count chart. The full run used the saved source test folder only to build the split manifest; it did not score test examples.

The pilot is a pipeline check. Do not use it to select a final model. These values are validation results, not test results. They do not settle the team's model choice. After the team freezes its model and settings, run the separate evaluator. Put reviewed final-test evidence in `final_test/` only after that run.

No checkpoints, raw images, `.venv`, full manifests, or duplicate run folders are stored here. The checkpoints and full run records remain in local ignored `outputs/runs/`. Export a run ZIP before a temporary Colab runtime ends.
