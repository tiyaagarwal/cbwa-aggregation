# Paired significance tests: CBWA vs FedAvg (final round-200 weights), run 27 Oct 2026 on Kaggle
(source: significance_tests.py; per-patient data in /kaggle/working/fedcb_results/significance_cbwa_vs_fedavg.json)

| Metric      | CBWA   | FedAvg | Diff    | 95% bootstrap CI of diff (patients resampled within hospital, 2000 reps) |
|-------------|--------|--------|---------|------------------|
| In-net F1   | 0.5453 | 0.4408 | +0.1045 | [+0.0289, +0.2048] |
| In-net Dice | 0.7690 | 0.6608 | +0.1082 | [+0.0993, +0.1175] |
| AU F1       | 0.8837 | 0.8837 | +0.0000 | [-0.0763, +0.0728] |
| AU Dice     | 0.8012 | 0.6993 | +0.1019 | [+0.0720, +0.1251] |
| TK F1       | 0.4520 | 0.4278 | +0.0242 | [-0.2720, +0.2054] |
| TK Dice     | 0.7590 | 0.6228 | +0.1361 | [+0.1039, +0.1819] |

Per hospital (16): CBWA F1 higher at 7, lower at 8 (1 tie); Wilcoxon p = 0.5321
Per hospital (16): CBWA Dice higher at 16, lower at 0; Wilcoxon p < 0.0001 (printed 0.0000)
In-network patients (n=215): tumour type correct CBWA 132, FedAvg 134; only-CBWA-right 22, only-FedAvg-right 24; McNemar p = 0.8830
   Dice: CBWA higher for 206/215, mean diff +0.1081; Wilcoxon p = 9.36e-34
External AU+TK (n=36): type correct CBWA 24, FedAvg 25; only-CBWA-right 2, only-FedAvg-right 3; McNemar p = 1.0000
   Dice: CBWA higher for 34/36, mean diff +0.1133; Wilcoxon p = 1.25e-07

# Patients per hospital (processed/federation.json + meta.json)
Site role     total train val  EP DIPG MB PA
ST training 246 197 49 31 59 72 84
SE training 180 144 36 31 36 78 35
CG training 112 90 22 7 54 44 7
UT training 96 77 19 13 15 31 37
IN training 88 70 18 4 13 45 26
CP training 72 58 14 17 0 33 22
TO training 69 54 15 18 0 48 3
GO training 58 47 11 16 12 20 10
PH training 41 32 9 8 13 13 7
AU external 24 0 0 0 24 0 0
DY training 21 16 5 3 3 10 5
NY training 19 15 4 4 8 7 0
DU training 18 14 4 0 0 18 0
BO training 14 11 3 0 0 14 0
TK external 12 0 0 2 1 7 2
CH training 10 7 3 0 4 2 4
TM training 9 7 2 0 0 3 6
KC training 2 1 1 0 0 2 0
total 1091 | EP 154, DIPG 242, MB 447, PA 248

# FedProx_pretrained200 started 27 Oct 2026 ~06:00 UTC in background (log /kaggle/working/fedprox_log.txt), warm-up rounds ~65 s

# Base study's released models (Lee et al. 2024; HF edhlee/FLPedBrain-processed checkpoints/), evaluated 27 Oct 2026
# with the same evaluation code and the same AU/TK patients as CBWA. Both loaded exactly (374/374 tensors).
| Model                         | AU F1  | AU Dice | TK F1  | TK Dice | Paired Dice test vs CBWA |
|-------------------------------|--------|---------|--------|---------|--------------------------|
| Authors' FL (fl_end_weights)  | 0.8837 | 0.8111  | 0.5476 | 0.7351  | AU: CBWA higher 19/24, p=0.095; TK: CBWA higher 6/12, p=0.30 |
| Authors' centralised (cds_end)| 1.0000 | 0.9374  | 1.0000 | 0.9362  | CBWA lower on all patients (AU p=1.2e-7, TK p=4.9e-4) |
| CBWA (ours, 16 hospitals)     | 0.8837 | 0.8012  | 0.4520 | 0.7590  | – |
Caveats: the authors' models were trained on 19 hospitals incl. TU + normal controls; whether AU/TK were excluded
from the released checkpoints' training data (especially the centralised one, perfect F1) cannot be verified from the release.
Their FL model's Dice under OUR evaluation (0.811/0.735) is lower than the paper's reported 0.845/0.818 -> metric/processing differences.
