# Results

The numbers reported in the paper, from the Kaggle runs (2 × NVIDIA T4). Each method was trained once.
Table and figure numbers refer to the paper (Results in Engineering version). These files contain scores
only: no MRI data, masks or model weights.

## Main study: pretrained I3D, 200 rounds (`main_200_rounds/`)

Rounds 1–50 warm-up on ST and SE (FedAvg), rounds 51–200 all 16 hospitals. FedAvg and CBWA use the same
class-balanced loss; FedProx uses plain cross-entropy as published. Code: `src/train_pretrained_baselines.py`.

| Metric (round 200) | FedAvg (+ CB loss) | FedProx | CBWA |
|---|---|---|---|
| In-network macro-F1 | 0.441 | 0.646 | 0.545 |
| In-network Dice | 0.661 | 0.717 | 0.769 |
| AU macro-F1 | 0.884 | 0.933 | 0.884 |
| AU Dice | 0.699 | 0.762 | 0.801 |
| TK macro-F1 | 0.428 | 0.731 | 0.452 |
| TK Dice | 0.623 | 0.702 | 0.759 |

| File | Paper | Contents |
|---|---|---|
| `final_scores.csv` | Table 7, Fig. 4 | The table above, unrounded |
| `fedprox_round200.json` | Table 7 | FedProx scores at round 200 |
| `significance_tests.md` | Table 8, Table 4 | Bootstrap intervals, Wilcoxon and McNemar tests (CBWA vs FedAvg); patients per hospital |
| `per_hospital.csv` | Table 9, Fig. 5 | F1 and Dice per hospital |
| `per_class.csv` | Table 10, Fig. 6 | F1 and correctly classified patients per tumor type |
| `reference_models_external.csv` | Table 11, Fig. 7 | Our models and the reference study's released models on AU and TK |
| `aggregation_weights_final_model.csv` | Fig. 8 | FedAvg and CBWA weight per hospital from the final model's validation F1 |
| `fig9_two_tk_patients.csv` | Fig. 9 | Predicted type, probability and Dice for the two TK patients |

## Preliminary study: model trained from scratch, 50 rounds (`preliminary_50_rounds/`)

Rounds 1–20 warm-up on ST and SE, rounds 21–50 all 16 hospitals. Code: `src/train.py`.

| Method (round 50) | In-net F1 | In-net Dice | AU F1 | TK F1 |
|---|---|---|---|---|
| FedAvg | 0.214 | 0.474 | 0.154 | 0.096 |
| FedProx | 0.344 | 0.570 | 0.000 | 0.177 |
| CBWA | 0.328 | 0.479 | 0.000 | 0.184 |

AU diagnostic (Table 14; `scripts/diagnose_external_site.py`), probability on the true class DIPG over the 24 AU patients:

| Method | Predicted class (argmax) | Mean P(DIPG) |
|---|---|---|
| FedAvg | MB 13, PA 9, DIPG 2 | 0.247 |
| FedProx | MB 23, PA 1, DIPG 0 | 0.116 |
| CBWA | MB 23, PA 1, DIPG 0 | 0.219 |

| File | Paper | Contents |
|---|---|---|
| `per_round_scores.csv` | Table 13, Fig. 10 | Scores every 5 rounds for the three methods |
| `kaggle_output/*.json` | Table 13, Fig. 10 | Raw per-round output saved by `src/train.py` on Kaggle (`FedAvg_history.json` is an earlier copy that stops at round 25; `FedAvg_results.json` is the full run) |

## Dataset

`dataset_patients_per_hospital.csv` (Table 4, Fig. 3): training and validation/test patients and tumor types
at each of the 18 hospitals; 1,091 patients in total.
