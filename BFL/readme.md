# Bayesian Federated Learning with Information-Geometric Barycenters and Cost-Free Personalization

This repository contains code to reproduce results from:
1. **Information-Geometric Barycenters for Bayesian Federated Learning** [1]
2. **Cost-Free Personalization via Information-Geometric Projection in Bayesian Federated Learning** [2]

## Mapping paper terms to code
- The algorithm **BA-BFL** from [1] corresponds to running:
  - `--alg=BFLAVG` with aggregation via `--update_method='wb_diag'` (Wasserstein barycenter) or
  - `--alg=BFLAVG` with aggregation via `--update_method='rkl'` (RKL barycenter; denoted RKLB in the paper).

- This codebase also supports training the **global and local models** needed for [2].
  To generate those models, set `--optimizer='ivon'` during training, then run the personalization pipeline in `Personalization/`
  (see `Personalization/README.md`).



# Requirements and Usage
## Requirements 
 You can install the requirements for this project by using requirements.txt
 ```sh 
  pip install -r requirements.txt 
 ```

## Data
### Data preperation
 The dataset is automatically downloaded and prepared by the code when first time running the experiment. For precaching the dataset, you can run the following command:

| $\textbf{Datasets}$ |     $\textbf{Image Size}$ | $\textbf{Number of Labels}$ | $\textbf{Train Size}$ | $\textbf{Test Size}$ |
|-------------------|------------------------:|:-----------------:|:-------------------:|:------------------:|
| FMNIST            | $1 \times 28 \times 28$ |        $10$       |       $60000$       |       $10000$      |
| Cifar-10          | $3 \times 32 \times 32$ |        $10$       |       $50000$       |       $10000$      |
| SVHN              | $3 \times 32 \times 32$ |        $10$       |       $73257$       |       $26032$      |

```sh
 $ python utils/data/data_downloader.py
```

### Non-IID Data Generation
As in the "Parallel Federated Learning Framework", we inherit the non-IID data generation methods from [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/pdf/2102.02079). You can run the following experiments:

| Experiment | Description |
|------------|-------------|
| IID        | IID data generation for 10 clients |
| IID-500    | IID data generation for 100 clients |
| noniid-labeldir    | Non-IID data generation for 10 clients with dirichlet distribution |
| noniid-labeldir-500    | Non-IID data generation for 100 clients with dirichlet distribution |
| noniid-label[1:4]    | Non-IID data generation for 10 clients with selection of how many class each client have (choices: 1, 2, 3, 4) |
| iid-diff-quantity    | IID data generation for 10 clients with different quantity of data |
| iid-diff-quantity-500    | IID data generation for 100 clients with different quantity of data |

The experiments described in the paper 'Barycentric Aggregationn for Bayesian Federated Learning' are based on the 'noniid-labeldir' experiment. 

## Usage
 You can run the experiments by using the following command:

```sh
 python train.py \
   --dataset=fmnist \
   --alg=BFLAVG \
   --experiment='noniid-labeldir' \
   --partition='noniid-labeldir' \
   --device='cuda' \
   --process=10 \
   --datadir='./data/' \
   --logdir='./logs' \
   --init_seed=0 \
   --optimizer='sgd' \
   --update_method = 'rkl' \ 
   --nbl = 2 
```

Running ``` fmnist_commands.sh ``` file reproduces the results on FashionMNIST. To reproduce the results on CIFAR-10 (SVHN), you only need to change the option dataset=fmnist by dataset=cifar10 (dataset=svhn) respectively in the commands written in the ``` fmnist_commands.sh ``` file.  


| Parameter | Description |
| ------ | ------ |
| dataset | Dataset name: cifar10, fmnist, kmnist, cifar100, svhn, covertype|
| alg | Algorithm name: BFL, BFLAVG, Fed, FedAVG|
| experiment | Experiment name: noniid-labeldir[-500], iid[-500], noniid-label[1:4], iid-dif-quantity[-500]|
| partition | Experiment name: noniid-labeldir[-500], iid[-500], noniid-label[1:4], iid-dif-quantity[-500]|
| device | Device name: cuda:0, cpu, mps|
| process | Number of processes for multiprocessing |
| datadir | Data directory path |
| logdir | Log directory path |
| init_seed | Initial seed number for the experiment |
| desc | Description of the experiment |
| optimizer | Optimizer used: sgd |
| update_method | The aggregation method: eaa, gaa, aalv, rkl (denoted RKLB in the paper), wb_diag (denoted WB in the paper) |
| nbl | The number of Bayesian layers used in the architecture (1,2,3) when the algorithm is Bayesian. By default, it is set to 3 |  



# Reference 
This code is based on the "Parallel Federated Learning Framework" https://github.com/ituvisionlab/BFL-P.git including the implementation of [How to Combine Variational Bayesian Networks in Federated Learning](https://arxiv.org/abs/2206.10897). We acknowledge the efforts of the contributors to this framework. 