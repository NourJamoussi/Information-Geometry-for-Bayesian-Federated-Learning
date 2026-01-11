import torch
import os
import logging
import numpy as np
import copy
import csv
import random
import pickle
from torch.utils.data import Subset, DataLoader
import ivon 
import pandas as pd 

from utils.fl import init_nets
from utils.update import personalized_global, personalized_global_ivon
from arg_parser import get_args
from utils.scores import compute_scores
from utils.ivon_utils import IVON_SAMP


def main(args, logger):
    global_model, nets = init_nets(args)
    networks = {"global_model": global_model, "nets": nets}
    personalized_networks = {lam : copy.deepcopy(networks["nets"]) for lam in args.lagrangian_parameters}
    loc_opt = {net_i: None for net_i in range(args.n_parties)}
    optimizers = {"global_optimizer": None, "local_optimizers": loc_opt}
    personalized_optimizers = {lam : copy.deepcopy(optimizers["local_optimizers"])  for lam in args.lagrangian_parameters}

    local_results_on_local_data = {net_id : [] for net_id in networks["nets"].keys()}
    local_results_on_global_data = {net_id : [] for net_id in networks["nets"].keys()}
    personalized_results_on_local_data = {lam :  {net_id : [] for net_id in networks["nets"].keys()} for lam in args.lagrangian_parameters}
    personalized_results_on_global_data = {lam :  {net_id : [] for net_id in networks["nets"].keys()} for lam in args.lagrangian_parameters}
    global_results_on_local_data = {net_id : [] for net_id in networks["nets"].keys()}
    global_results = []

    
    dir = f'logs/ivon/{args.dataset}/{args.init_seed}'
    experiment_dir = f'{dir}/{args.alg}/{args.update_method}'
    results_dir = f'{experiment_dir}/results'
    models_dir = f'{experiment_dir}/models'
    perso_models_dir = f'{models_dir}/personalized/{args.perso_method}/'
    optimizers_dir = f'{experiment_dir}/optimizers'
    perso_optimizers_dir = f'{optimizers_dir}/personalized/{args.perso_method}/'

    # Load the global and local models
    global_model_path = f"{models_dir}/checkpoint.pt"
    local_model_paths = [f"{models_dir}/client_{net_id}.pt" for net_id in networks["nets"].keys()]
    global_model.load_state_dict(torch.load(global_model_path, map_location=args.device))
    for net_id, local_model_path in zip(networks["nets"].keys(), local_model_paths):
        networks["nets"][net_id].load_state_dict(torch.load(local_model_path, map_location=args.device))

    # Load the optimizers 
    if args.optimizer == 'ivon':
        # Load the local optimizers
        for net_id in networks["nets"].keys():
            optimizer = ivon.IVON(networks["nets"][net_id].parameters(), lr=args.lr, ess=1, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init)
            optimizer.load_state_dict(torch.load(f"{optimizers_dir}/client_{net_id}_optimizer.pt", map_location=args.device))
            optimizers["local_optimizers"][net_id] = optimizer
        # Load the global optimizer
        optimizer = IVON_SAMP(networks["global_model"].parameters(), lr=args.lr, ess=1, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init, cov=None)
        optimizer.load_state_dict(torch.load(f"{optimizers_dir}/global_optimizer.pt", map_location=args.device))
        optimizers["global_optimizer"] = optimizer
    else: 
        pass 

    ############################# Personalization ###############################

    # Personalization of local models 
    logger.critical(f"Personalization of models")
    if args.optimizer == 'ivon':
        for lam in args.lagrangian_parameters:
            personalization_weight = lam / (lam + 1)
            weights = [personalization_weight, 1 - personalization_weight]
            for net_id in networks["nets"].keys():
                projected_net_dict, cov_agg = personalized_global_ivon(args, networks, net_id, optimizers, personalization_weight)
                personalized_networks[lam][net_id].load_state_dict(projected_net_dict)
                personalized_optimizers[lam][net_id] = IVON_SAMP(personalized_networks[lam][net_id].parameters(), lr=args.lr, ess=1, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init, cov=cov_agg)
                # Save the personalized model and optimizer
                os.makedirs(f"{perso_models_dir}/{lam}", exist_ok=True)
                torch.save(projected_net_dict, f"{perso_models_dir}/{lam}/perso_client_{net_id}.pt")
                os.makedirs(f"{perso_optimizers_dir}/{lam}", exist_ok=True)
                torch.save(personalized_optimizers[lam][net_id].state_dict(), f"{perso_optimizers_dir}/{lam}/perso_client_{net_id}_optimizer.pt")
    else: 
        for lam in args.lagrangian_parameters:
            personalization_weight = lam / (lam + 1)
            for net_id in networks["nets"].keys():
                projected_net_dict = personalized_global(args, networks, net_id, personalization_weight)
                personalized_networks[lam][net_id].load_state_dict(projected_net_dict)
                os.makedirs(f"{perso_models_dir}/{lam}", exist_ok=True)
                torch.save(projected_net_dict, f"{perso_models_dir}/{lam}/perso_client_{net_id}.pt")


    ############################## Loading the test data ##############################

    # Read the test_dl_global pickle file
    with open(f'{dir}/test_dl_global_{args.init_seed}.pkl', 'rb') as f:
        test_dl_global = pickle.load(f)
    test_dataset = test_dl_global.dataset
    # Read the train_data_statistics pickle file
    with open(f'{dir}/train_data_statistics_{args.init_seed}.pkl', 'rb') as f:
        train_data_statistics = pickle.load(f)

    # Create the test data for each client following the train data distribution
    # A dictionary to store the count of samples per class
    class_counts = {}
    # Iterate over the dataset
    for _, class_label in test_dataset:
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1
    class_counts = dict(sorted(class_counts.items()))
    K = len(class_counts) #number of classes in the dataset
    N = args.n_parties #number of clients
    # A dictionary to store the proportions of each class in the train data of each client
    proportions = {}
    for net_id in range(N):
        proportions[net_id] = {label: 0 for label in range(K)}
    for net_id, classes in train_data_statistics.items():
        total_sum = sum(classes.values())
        for class_id, value in classes.items():
            proportions[net_id][class_id] = value / total_sum 
    # A dictionary to store the number of test samples per class for each client
    frequencies = {}
    for net_id in range(N):
        frequencies[net_id] = {label: round(proportions[net_id][label] * class_counts[label]) for label in range(K)} 
    # A dictionary to store indices per class 
    indices_dict = {}
    for index, (data_point, class_label) in enumerate(list(test_dataset)):
        if class_label in indices_dict:
            indices_dict[class_label].append(index)
        else:
            indices_dict[class_label] = [index]
    indices_dict = dict(sorted(indices_dict.items()))
    # Local test data loaders
    test_local_dataloaders = []
    for net_id in range(N):
        client_frequencies = frequencies[net_id]
        client_indices = []
        for class_id, count in client_frequencies.items():
            class_indices = indices_dict[class_id]
            client_indices.extend(class_indices[:count])
        client_dataset = Subset(test_dl_global.dataset, client_indices)
        client_dataloader = DataLoader(client_dataset, batch_size=32, shuffle=False)
        test_local_dataloaders.append(client_dataloader)


    ################################################## EVALUATION ###################################################################


    logger.critical(f"Evaluation of local models")
    ############### Evaluation of local models on local data ###############
    #if os.path.exists(f"{results_dir}/local_on_local/client_{net_id}_on_local_data.csv"):
    #    logger.critical(f"Already Evaluated")
    #else:
    logger.critical(f"Evaluation of local models on the local test data")
    for net_id in networks["nets"].keys():
        test_dl_local= test_local_dataloaders[net_id]
        if args.optimizer == 'ivon':
            optimizer = optimizers["local_optimizers"][net_id]
        else:
            optimizer = None
        test_acc, test_ece, test_nll = compute_scores(networks["nets"][net_id], test_dl_local, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
        networks["nets"][net_id].cpu()
        logger.info(f'>> Model {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
        local_results_on_local_data[net_id].append((test_acc, test_ece, test_nll))
    ############### Evaluation of local models on global data ###############
    #if os.path.exists(f"{results_dir}/local_on_global/client_{net_id}_on_global_data.csv"):
    #    logger.critical(f"Already Evaluated")    
    #else:
    logger.critical(f"Evaluation of local models on the global test data")
    for net_id in networks["nets"].keys():
        if args.optimizer == 'ivon':
            optimizer = optimizers["local_optimizers"][net_id]
        else:
            optimizer = None
        test_acc, test_ece, test_nll = compute_scores(networks["nets"][net_id], test_dl_global, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
        networks["nets"][net_id].cpu()
        logger.info(f'>> Model {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
        local_results_on_global_data[net_id].append((test_acc, test_ece, test_nll))


    logger.critical(f"Evaluation of the global model")
    ############### Evaluation of the global model on the Local data ###############
    logger.critical(f"Evaluation of the global model on the local test data")
    if args.optimizer == 'ivon':
        optimizer = optimizers["global_optimizer"]
    else:
        optimizer = None
    for net_id in networks["nets"].keys():
        test_dl_local= test_local_dataloaders[net_id]
        test_acc, test_ece, test_nll = compute_scores(networks["global_model"], test_dl_local, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
        networks["global_model"].cpu()
        logger.info(f'>> Global Model on the local data of client {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
        global_results_on_local_data[net_id].append((test_acc, test_ece, test_nll))
    ############### Evaluation of the global model on the Global data ###############
    #if os.path.exists(f"{results_dir}/global_on_global.csv"):
    #    logger.critical(f"Already Evaluated")    
    #else:
    logger.critical(f"Evaluation of the global model on the global data")        
    test_acc, test_ece, test_nll = compute_scores(networks["global_model"], test_dl_global, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
    networks["global_model"].cpu()
    logger.info(f'>> Global Model Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
    global_results.append((test_acc, test_ece, test_nll))

    logger.critical(f"Evaluation of the personalized models")
    ############### Evaluation of Personalized models on Local data ############### #TODO for IVON
    for lam in args.lagrangian_parameters:
        logger.critical(f"Lambda: {lam}")
        logger.critical(f"Evaluation of personalized models on the local test data")
        for net_id in networks["nets"].keys():
            test_dl_local= test_local_dataloaders[net_id]
            if args.optimizer == 'ivon':
                optimizer = personalized_optimizers[lam][net_id]
            else:
                optimizer = None
            test_acc, test_ece, test_nll = compute_scores(personalized_networks[lam][net_id], test_dl_local, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
            networks["nets"][net_id].cpu()
            logger.info(f'>> Personalized Model {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
            personalized_results_on_local_data[lam][net_id].append((test_acc, test_ece, test_nll))
        ############### Evaluation of Personalized models on Global data ############### #TODO for IVON
        logger.critical(f"Evaluation of personalized models on the global test data")
        for net_id in networks["nets"].keys():
            if args.optimizer == 'ivon':
                optimizer = personalized_optimizers[lam][net_id]
            else:
                optimizer = None
            test_acc, test_ece, test_nll = compute_scores(personalized_networks[lam][net_id], test_dl_global, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
            networks["nets"][net_id].cpu()
            logger.info(f'>> Personalized Model {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
            personalized_results_on_global_data[lam][net_id].append((test_acc, test_ece, test_nll))

    networks["global_model"].to("cpu")


    # Ensure the directories exist before writing files
    os.makedirs(f"{results_dir}/local_on_global", exist_ok=True)
    os.makedirs(f"{results_dir}/local_on_local", exist_ok=True)
    os.makedirs(f"{results_dir}/global_on_local", exist_ok=True)

    # Save the results of the global model on the global data
    with open(f"{results_dir}/global_on_global.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Acc", "ECE", "NLL"))
        writer.writerows(global_results)

    for net_id in networks["nets"].keys():
        # Save the results of the global model on the local data
        with open(f"{results_dir}/global_on_local/global_on_local_data_{net_id}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(("Acc", "ECE", "NLL"))
            writer.writerows(global_results_on_local_data[net_id])     
        # Save the results of the local models on the global data
        with open(f"{results_dir}/local_on_global/client_{net_id}_on_global_data.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(("Acc", "ECE", "NLL"))
            writer.writerows(local_results_on_global_data[net_id])
        # Save the results of the local models on the local data
        with open(f"{results_dir}/local_on_local/client_{net_id}_on_local_data.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(("Acc", "ECE", "NLL"))
            writer.writerows(local_results_on_local_data[net_id])
    
    for lam in args.lagrangian_parameters:
        # Ensure the directories exist before writing files
        perso_results_dir = f"{results_dir}/personalized/{args.perso_method}/{lam}"
        os.makedirs(f"{perso_results_dir}/perso_on_global", exist_ok=True)
        os.makedirs(f"{perso_results_dir}/perso_on_local", exist_ok=True)
        for net_id in networks["nets"].keys():
            # Save the results of the personalized model on the global data
            with open(f"{perso_results_dir}/perso_on_global/perso_{net_id}_on_global_data.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(("Acc", "ECE", "NLL"))
                writer.writerows(personalized_results_on_global_data[lam][net_id])
            # Save the results of the personalized model on the local data
            with open(f"{perso_results_dir}/perso_on_local/perso_{net_id}_on_local_data.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(("Acc", "ECE", "NLL"))
                writer.writerows(personalized_results_on_local_data[lam][net_id])
   

if __name__ == '__main__':
    args = get_args()

    dir = f'logs/ivon/{args.dataset}/{args.init_seed}'
    experiment_dir = f'{dir}/{args.alg}/{args.update_method}'
    results_dir = f'{experiment_dir}/results'
    args.logdir = f"{results_dir}/personalized/{args.perso_method}"

    os.makedirs(f"{args.logdir}", exist_ok=True)

    
    args.device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=f'{args.logdir}/log.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M-%S', level=logging.DEBUG, filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(args)
    logger.info(f"device: {args.device}")

    seed = args.init_seed
    logger.info("#" * 100)

    if args.desc != "":
        logger.critical(f"Description: {args.desc}")
    if args.alg == "BFLAVG":
        logger.critical(f"Optimizer: {args.optimizer}")
        logger.critical(f"Update Method: {args.update_method}")
        if args.arch in ["bcnn", "bcnn_1", "bcnn_2", "bfcnn"]:
            logger.critical(f"Number of Bayesian Layers: {args.nbl}")
    logger.critical(f"The Lagrangian parameters used for personalization: {args.lagrangian_parameters}")
    logger.critical(f"Personalization method: {args.perso_method}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    main(args, logger)


    def merge_results(args):
        # Define directories
        dir = f'logs/ivon/{args.dataset}/{args.init_seed}'
        experiment_dir = f'{dir}/{args.alg}/{args.update_method}'
        results_dir = f'{experiment_dir}/results'

        # Initialize an empty DataFrame to store all results
        merged_data = pd.DataFrame()

        # Helper function to read and append CSV files
        def read_and_append_csv(folder, label):
            nonlocal merged_data
            folder_path = os.path.join(results_dir, folder)
            if os.path.isfile(folder_path + '.csv'):
                # If the path is a file, read it directly
                df = pd.read_csv(folder_path + '.csv')
                df['source'] = label
                df['file'] = folder + '.csv'
                df['file_number'] = 0  # Assign a default number for single files
                merged_data = pd.concat([merged_data, df], ignore_index=True)
            else:
                # If the path is a directory, read all CSV files in it
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(folder_path, file_name)
                        df = pd.read_csv(file_path)
                        df['source'] = label
                        df['file'] = file_name
                        # Extract the number from the file name
                        file_number = int(''.join(filter(str.isdigit, file_name)))
                        df['file_number'] = file_number
                        merged_data = pd.concat([merged_data, df], ignore_index=True)

        # Read global model results
        read_and_append_csv('global_on_local', 'global_on_local')
        read_and_append_csv('local_on_global', 'local_on_global')
        read_and_append_csv('global_on_global', 'global_on_global')
        read_and_append_csv('local_on_local', 'local_on_local')

        # Read personalized model results
        for lam in args.lagrangian_parameters:
            read_and_append_csv(f'personalized/{args.perso_method}/{lam}/perso_on_local', f'perso_on_local_lambda_{lam}')
            read_and_append_csv(f'personalized/{args.perso_method}/{lam}/perso_on_global', f'perso_on_global_lambda_{lam}')

        # Sort the merged data by source and file_number
        merged_data.sort_values(by=['source', 'file_number'], inplace=True)

        # Drop the file_number column as it's no longer needed
        merged_data.drop(columns=['file_number'], inplace=True)

        # Save the merged data to a new CSV file
        merged_data.to_csv(os.path.join(f'{results_dir}/personalized/{args.perso_method}/', 'merged_results.csv'), index=False)

    merge_results(args)