# CBWA: training and evaluation pipeline

Code for the experiments in:

> T. Agarwal, C. Patil and S. Saman, *Class-Balance-Aware Weighted Aggregation for Federated Pediatric Brain Tumor MRI Classification and Segmentation*, Results in Engineering (manuscript submitted).

**What this repository is.** The code used to run the paper's experiments on Kaggle (2 × NVIDIA T4): data preparation, federated training of FedAvg, FedProx and CBWA with the pretrained I3D model, evaluation, significance tests and the demonstration that draws Fig. 9. The CBWA aggregation rule as it ran (Eqs. (4)–(5) of the paper, with the weight floor and momentum) is the `cbwa()` function in [`pipeline/src/train_pretrained_baselines.py`](pipeline/src/train_pretrained_baselines.py); a standalone version without momentum is in [`pipeline/src/fl_core/aggregation.py`](pipeline/src/fl_core/aggregation.py).

**Results.** The Kaggle results reported in the paper (final scores, significance tests, per-hospital and per-class results, and the preliminary 50-round runs) are in [`pipeline/results/`](pipeline/results/).

**No patient data.** The repository contains no MRI data, masks, model weights or access tokens. The data are available through gated access on Hugging Face ([`edhlee/FLPedBrain-processed`](https://huggingface.co/datasets/edhlee/FLPedBrain-processed)); the I3D model definition comes from the reference study's repository [`edhlee/FLPedBrain`](https://github.com/edhlee/FLPedBrain).

**How to run it.** Requirements, setup and a description of every script are in [`pipeline/README.md`](pipeline/README.md).

## How to cite

Please cite the paper above. To cite the code itself, use the metadata in [`CITATION.cff`](CITATION.cff) or the Zenodo DOI shown on the repository's release page.

## Licence

MIT; see [`LICENSE`](LICENSE).
