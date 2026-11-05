# Training pipeline

The code used to run the paper's experiments on Kaggle (2 × NVIDIA T4): data preparation,
federated training of FedAvg, FedProx and CBWA with the pretrained I3D model, evaluation,
significance tests and the live demonstration that draws Fig. 9.

This folder contains **code only**. It contains no MRI data, masks, model weights or access tokens.

## What you need

- **Data:** the FLPedBrain-processed release ([`edhlee/FLPedBrain-processed`](https://huggingface.co/datasets/edhlee/FLPedBrain-processed)
  on Hugging Face). Access is gated: request it from the dataset authors. The raw data are in the
  Stanford Digital Repository, doi: [10.25740/bf070wx6289](https://doi.org/10.25740/bf070wx6289).
- **Model code:** a clone of the authors' repository [`edhlee/FLPedBrain`](https://github.com/edhlee/FLPedBrain) in the
  working folder, and the ImageNet + Kinetics I3D weights `rgb_imagenet.pt`. The scripts import the model
  definition `model_pretrained.py` from `FLPedBrain/FLPedBrain-PyTorch/src`; that file is not included here.
- **Packages:** `pip install -r requirements.txt`.

The scripts expect everything under `/kaggle/working`; set `FEDCB_ROOT` to use another folder.

## Contents

| Path | What it does |
|---|---|
| `scripts/00_probe_one_file.py` | Downloads one small hospital file to check the data format |
| `scripts/preprocess.py` | Converts each hospital's file into volumes, masks and labels |
| `scripts/build_federation_real.py` | Patient-level train/validation split (seed 0) and class weights |
| `src/data/` | Data loading, preprocessing and federation helpers |
| `src/fl_core/` | Aggregation rules (FedAvg, FedProx, CBWA) and losses |
| `src/train.py` | Preliminary study: model trained from scratch, 50 rounds |
| `src/train_pretrained_baselines.py` | Main study: pretrained I3D, 200 rounds, CBWA and published baselines |
| `src/significance_tests.py` | Paired tests of CBWA vs FedAvg on the same patients |
| `src/authors_models_eval.py` | Evaluates the reference study's released models on AU and TK |
| `src/demo_inference.py` | Live demonstration and Fig. 9 (`Figure_9.pdf`) |
| `scripts/diagnose_external_site.py` | Per-class probabilities on the external hospital AU (Table 14) |
| `tests/test_cbwa.py` | Checks the CBWA rule inside a training script: `python tests/test_cbwa.py src/train_pretrained_baselines.py` |
| `tests/make_mock_federation.py` | Small synthetic federation for running the scripts on CPU |

`src/authors_models_eval.py` reads a Hugging Face token from a Kaggle secret named `HF_TOKEN` at run time; no token is stored here.

## License

MIT, see [LICENSE](LICENSE).
