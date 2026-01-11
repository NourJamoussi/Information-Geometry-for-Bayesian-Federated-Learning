# Repository Overview

This repository contains the official code to reproduce the results of the following papers:
1. **Information-Geometric Barycenters for Bayesian Federated Learning**
2. **Cost-Free Personalization via Information-Geometric Projection in Bayesian Federated Learning**

## Structure
The repository is organized into two main folders:
- `BFL/`
- `Personalization/`

### `BFL/`
Contains the code for Paper (1) and the federated learning training needed to reproduce Paper (2).  
Running the code under `BFL/` produces training artifacts (e.g., checkpoints and logs) that are used by the personalization step.

### `Personalization/`
Contains the code to reproduce the final results of Paper (2), **after** you obtain the required artifacts from `BFL/` (global/local models and training-data statistics, extracted from the BFL logs/checkpoints).

## How to use this repo
1. Run the federated learning training in `BFL/` to reproduce Paper 1 results and generate checkpoints and input required to reproduce Paper 2 results.
2. Extract the required artifacts from the `BFL/` outputs (see `BFL/README.md`).
3. Run the personalization pipeline in `Personalization/` (see `Personalization/README.md`).

Each folder contains its own README with detailed instructions and explanations of the code and folder structure.



## Citation
If you use this code or results in your research, please cite:

```bibtex
@article{jamoussi2024information,
  title={Information-Geometric Barycenters for Bayesian Federated Learning},
  author={Jamoussi, Nour and Serra, Giuseppe and Stavrou, Photios A and Kountouris, Marios},
  journal={arXiv preprint arXiv:2412.11646},
  year={2024}
}

@article{jamoussi2025cost,
  title={Cost-Free Personalization via Information-Geometric Projection in Bayesian Federated Learning},
  author={Jamoussi, Nour and Serra, Giuseppe and Stavrou, Photios A and Kountouris, Marios},
  journal={arXiv preprint arXiv:2509.10132},
  year={2025}
}



