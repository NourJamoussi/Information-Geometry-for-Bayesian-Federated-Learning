import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fmnist', choices=["cifar10", "fmnist", "kmnist", "cifar100", "svhn", "covertype", "speechcommands"], help='dataset used for training')
    parser.add_argument('--alg', type=str, default='FedAVG', choices=["BFL", "BFLAVG", "Fed", "FedAVG", "FedProx", "FedNova", "Scaffold"], help='communication strategy')
    parser.add_argument('--n_parties', type=int, default=10, help='the number of clients')
    parser.add_argument('--is_same_initial', type=int, default=1, help='whether initial all the models with the same parameters')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--device', type=str, default='mps', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='ivon', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.01, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--update_method', type=str, default="none", choices=["eaa","wb_diag","rkl","gaa","aalv"], help='aggregation method for Bayesian FL')
    parser.add_argument('--desc', type=str, default="", help='Description of run')
    
    parser.add_argument('--nbl', type=int, default=0, choices=[0,1,2,3], help='Number of Bayesian layers in the architecture')
    parser.add_argument('--n_samples', type=int, default=10,  help='Number of Samples during the testing when the algorithm is bayesian')

    parser.add_argument('--lagrangian_parameters', type=list, default=[1/4,1/2,1,3/2,2], help='The Lagrangian parameters tested for personalization') 
    parser.add_argument('--perso_method', type=str, default="none", choices=["wb_diag","rkl","none"], help='personalization method for Bayesian FL')

    parser.add_argument('--hess_init', type=float, default=0.01, help='Parameter initializing the hessian in IVON optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.01)') #LR = 0.01 for SGD - 0.1 for IVON  , won't be used, just matter of initialization
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength") # won't be used, just matter of initialization


    args = parser.parse_args()
    

    if args.optimizer == "ivon":
        if args.dataset == "speechcommands":
            args.arch = "cnn_speech"
        elif args.dataset == "covertype":
            args.arch = "fcnn"
        else:
            args.arch = "cnn"
    else: 
        if args.alg == "FedAVG" or args.alg == "Fed":
            args.arch = "cnn"
        elif args.alg == "BFL" or args.alg == "BFLAVG":
            if args.nbl == 1:
                args.arch = "bcnn_1"
            elif args.nbl == 2:
                args.arch = "bcnn_2"
            elif args.nbl == 3: 
                args.arch = "bcnn"
    
        if args.dataset == "covertype":
            if args.alg == "FedAVG" or args.alg == "Fed":
                args.arch = "fcnn"
            elif args.alg == "BFL" or args.alg == "BFLAVG":
                args.arch = "bfcnn"
        
        if args.dataset == "speechcommands":
            if args.alg == "FedAVG" or args.alg == "Fed":
                args.arch = "cnn_speech"
            elif args.alg == "BFL" or args.alg == "BFLAVG":
                args.arch = "bcnn_speech"
    return args
