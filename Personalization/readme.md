# Cost-Free Personalization via Information-Geometric Projection in Bayesian Federated Learning

This repository contains the official code to reproduce the results of the paper  
**"Cost-Free Personalization via Information-Geometric Projection in Bayesian Federated Learning"**.

## Overview
Given artifacts produced by Bayesian Federated Learning (BFL) training with the **IVON** optimizer, this code computes personalized models and compute the evaluation on local and global data.

## Inputs
This code expects the following as input (extracted from BFL training logs):
1. Global and local models
2. Global and local IVON optimizers
3. Training data statistics
4. Global test dataloader

## Outputs
It produces:
1. Personalized models
2. Personalized optimizers
3. Evaluation results of all models on local and global data

## Reproducing the results
1. **Run BFL training with IVON**  
   Use the script below to run all datasets:
   - `ivon_commands_all_datasets.sh`

2. **Prepare directories and extract required inputs from logs**  
   Run:
   - `directory_prep.ipynb`  
   This notebook organizes the `logs/` folder structure and extracts the inputs required by this repository from the BFL logs.

3. **Run personalization**
   Run:
   - `main.py`  
   Or use the script for all datasets:
   - `run_personalization_ivon.sh`

## Notes
- Ensure that the BFL training logs are available before running `directory_prep.ipynb`.
- Scripts are provided to reproduce results across multiple datasets.
