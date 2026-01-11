import torch
import os
import datetime 
import logging
import numpy as np
import time
import copy
import csv
import random
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.data.data import data_handler
from utils.FL.fl import init_nets
from utils.FL.utils import train_handler
from utils.ivon_utils import IVON_SAMP
from arg_parser import get_args
from algorithms import get_algorithm
from scores import compute_scores

def local_train(args, algo, networks, selected, net_dataidx_map):
    start = time.time()
    processes = []
    finished_process = 0

    for net_id, net in networks["nets"].items():
        if net_id not in selected:
            continue

        dataidxs = net_dataidx_map[net_id]
        net.dataset_size = len(dataidxs)
        process = mp.Process(target=algo.local_update, args=(args, networks["nets"][net_id], networks["global_model"], net_id, dataidxs))
        process.start()
        processes.append(process)

        if len(processes) == args.process:
            for p in processes:
                p.join()
                finished_process += 1
            processes = []
    for p in processes:
        p.join()
        finished_process += 1
    processes = []
    logger.info(f"{(time.time()-start):.2f} second.")


def main(args, logger, pre_selected):
    algo = get_algorithm(args)()
    test_dl_global, net_dataidx_map = data_handler(args, logger)
    global_model, nets = init_nets(args)
    networks = {"global_model": global_model, "nets": nets}

    local_results_on_global_data = {net_id : [] for net_id in networks["nets"].keys()}

    results = []

    args.round = 0

    for args.round in range(args.comm_round):        
        logger.info(f"Communication round: {args.round} / {args.comm_round}")
        # selection of clients
        if args.partition == "noniid-labeldir-500" or args.partition == "iid-500":
            selected = pre_selected[args.round]
        else:
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

        global_para = networks["global_model"].state_dict()
        if args.round == 0:
            if args.is_same_initial:
                for net_id in selected:
                    networks["nets"][net_id].load_state_dict(global_para)
        else:   
            for net_id in selected:
                    # update the mean 
                    networks["nets"][net_id].load_state_dict(global_para)
                    # Update the covariance 
                    if args.optimizer == 'ivon':
                        global_optimizer = IVON_SAMP(networks["nets"][net_id].parameters(), lr=args.lr, ess=1, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init, cov=None)
                        global_optimizer_file = f"{args.logdir}/global_optimizer.pt"
                        if os.path.isfile(global_optimizer_file):
                            global_optimizer.load_state_dict(torch.load(global_optimizer_file))
                            global_cov = global_optimizer.param_groups[0]["cov"]
                            train_dataloader , optimizer , _ = train_handler(args, networks["nets"][net_id], net_id, net_dataidx_map[net_id])
                            N = len(train_dataloader.dataset)
                            global_hessian = (1/ (N * global_cov)) - args.reg 
                            optimizer.load_state_dict(torch.load(f"{args.logdir}/clients/client_{net_id}_optimizer.pt"))
                            optimizer.param_groups[0]["hess"] = torch.from_numpy(global_hessian.astype(np.float32)).to(args.device)
                            torch.save(optimizer.state_dict(), f"{args.logdir}/clients/client_{net_id}_optimizer.pt")
                        else: 
                            print(f"Global optimizer file not found at {global_optimizer_file}")
            logger.info(f'>> Global Model Cov : Min: {np.min(global_cov)}, Max: {np.max(global_cov)}, Mean: {np.mean(global_cov)}, Std: {np.std(global_cov)}')

        # party executes
        if args.pretrained == False or (args.pretrained == True and args.round > 0):
            local_train(args, algo, networks, selected, net_dataidx_map)


        # updating parameters of trained networks via reading files due to process operations
        for net_id in networks["nets"].keys():
            if net_id not in selected or (args.alg.lower() == "fednova"):
                continue
            networks["nets"][net_id].load_state_dict(torch.load(f"{args.logdir}/clients/client_{net_id}.pt"))

        # global update rule
        algo.global_update(args, networks, selected, net_dataidx_map)




        ################################################## EVALUATION ###################################################################
        logger.critical(f"Evaluation of local models after round {args.round}")
        ############### Evaluation of local models on global data ###############
        logger.critical(f"Evaluation of local models after round {args.round} on the global test data")
        for net_id in networks["nets"].keys():
            if net_id not in selected or (args.alg.lower() == "fednova"):
                continue
            _ , optimizer , _ = train_handler(args, networks["nets"][net_id], net_id, net_dataidx_map[net_id])
            optimizer.load_state_dict(torch.load(f"{args.logdir}/clients/client_{net_id}_optimizer.pt"))
            test_acc, test_ece, test_nll = compute_scores(networks["nets"][net_id], test_dl_global, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
            networks["nets"][net_id].cpu()
            logger.info(f'>> Model {net_id} Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
            local_results_on_global_data[net_id].append((args.round + 1, test_acc, test_ece, test_nll))


        logger.critical(f"Evaluation of the global model after round {args.round}")
        ############### Evaluation of the global model on the Global data ###############
        logger.critical(f"Evaluation of the global model after round {args.round} on the global data")        
        test_acc, test_ece, test_nll = compute_scores(networks["global_model"], test_dl_global, args, device=args.device, optimizer=optimizer, n_sample=[1,args.n_samples]["bfl" in args.alg.lower()])
        networks["global_model"].cpu()
        logger.info(f'>> Global Model Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
        results.append((args.round + 1, test_acc, test_ece, test_nll))



    networks["global_model"].to("cpu")
    torch.save(networks["global_model"].state_dict(), f"{args.logdir}/checkpoint.pt")

    with open(f"{args.logdir}/global_log.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Round", "Acc", "ECE", "NLL"))
        writer.writerows(results)

    for net_id in local_results_on_global_data.keys():
        # Save the results of the local models on the global data
        with open(f"{args.logdir}/clients/local_results_{net_id}_on_global_data.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(("Round", "Acc", "ECE", "NLL"))
            writer.writerows(local_results_on_global_data[net_id])
  

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(1)
    args = get_args()

    if args.pretrained == False:
        log_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
        if args.arch in ["bcnn", "bcnn_1", "bcnn_2", "bfcnn"]:
            args.logdir = os.path.join(f"{args.logdir}/{args.dataset}/{args.partition}/{args.alg}/{args.update_method}/{args.init_seed}/{args.nbl}/{log_time}")
        else : 
            args.logdir = os.path.join(f"{args.logdir}/{args.dataset}/{args.partition}/{args.alg}/{args.update_method}/{args.init_seed}/{log_time}")
        os.makedirs(f"{args.logdir}/clients", exist_ok=True)
        for round_id in range(args.comm_round):
            os.makedirs(f"{args.logdir}/clients/round_{round_id}", exist_ok=True)

    else : 
        args.logdir = args.pretrained_logdir 
    
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pre_selected = {}
    if args.partition == "noniid-labeldir-500" or args.partition == "iid-500":
        for i in range(args.comm_round):
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            pre_selected[i] = arr[:int(args.n_parties * args.sample)]
        with open(f"{args.logdir}/pre_selected.txt", 'w') as f:
            for i in range(args.comm_round):
                f.write(f"{i}: {pre_selected[i]}\n")
    main(args, logger, pre_selected)