from architectures import CNN, BCNN, FcNN, BFcNN, BCNN_1, BCNN_2, BCNN_Speech, CNN_Speech

def init_nets(args):
    nets = {net_i: None for net_i in range(args.n_parties)}
    if args.dataset == 'cifar10':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'fmnist':
        input_channel = 1
        input_dim = (16 * 4 * 4)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'kmnist':
        input_channel = 1
        input_dim = (16 * 4 * 4)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'cifar100':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 100
    elif args.dataset == 'covertype':
        input_dim = 54
        hidden_dims=[100, 50]
        output_dim = 7
    elif args.dataset == 'svhn':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'speechcommands':
        input_dim = 1 
        output_dim = 35
        stride = 16 
        n_channel = 32 

    for net_i in range(args.n_parties):        
        if args.arch.lower() == "cnn":
            net = CNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == "bcnn":
            net = BCNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == "bcnn_1":
            net = BCNN_1(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == 'bcnn_2':
            net = BCNN_2(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == "fcnn":
            net = FcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        elif args.arch == "bfcnn":
            net = BFcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        elif args.arch == "cnn_speech":
            net = CNN_Speech(input_dim=input_dim, output_dim=output_dim, stride=stride, n_channel=n_channel) 
        elif args.arch == "bcnn_speech":
            net = BCNN_Speech(input_dim=input_dim, output_dim=output_dim, stride=stride, n_channel=n_channel) 
        else:
            raise ValueError("Unknown architecture: {}".format(args.arch))
        nets[net_i] = net


    if args.arch.lower() == "cnn":
        global_net = CNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
    elif args.arch == "bcnn":
        global_net = BCNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
    elif args.arch == "bcnn_1":
        global_net = BCNN_1(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
    elif args.arch == 'bcnn_2':
        global_net = BCNN_2(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)    
    elif args.arch == "fcnn":
        global_net = FcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    elif args.arch == "bfcnn":
        global_net = BFcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    elif args.arch == "cnn_speech":
        global_net = CNN_Speech(input_dim=input_dim, output_dim=output_dim, stride=stride, n_channel=n_channel) 
    elif args.arch == "bcnn_speech":
        global_net = BCNN_Speech(input_dim=input_dim, output_dim=output_dim, stride=stride, n_channel=n_channel) 
    else:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    if args.is_same_initial:
        global_para = global_net.state_dict() 
        for net_id, net in nets.items():
            net.load_state_dict(global_para)

    return global_net, nets
