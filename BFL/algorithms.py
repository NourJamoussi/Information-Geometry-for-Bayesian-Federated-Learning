import torch
import math
from utils.FL.utils import train_handler
from utils.FL.update import update_global
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

def plot_losses( losses, net_id, round_id, logdir):
    plt.plot( losses)
    plt.title(f"Losses of client {net_id} in round {round_id}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"{logdir}/clients/round_{round_id}/client_{net_id}_losses.png")
    plt.close()

def get_algorithm(args):
    if args.alg.lower() == "bfl":
        return BFL
    elif args.alg.lower() == "bflavg":
        return BFLAvg
    elif args.alg.lower() == "fed":
        return FED
    elif args.alg.lower() == "fedavg":
        return FEDAvg
    elif args.alg.lower() == "fedprox":
        return FEDProx
    elif args.alg.lower() == "fednova":
        return FEDNova
    elif args.alg.lower() == "scaffold":
        return Scaffold
    else:
        raise NotImplementedError(f"{args.alg} is not implemented!")

class BaseAlgorithm():
    def local_update(self, args, net, global_net, net_id, dataidxs):
        raise NotImplementedError() 
    
    def global_update(self, args, networks, selected, net_dataidx_map):
        raise NotImplementedError()


class BFL(BaseAlgorithm):
    def local_update(self, args, net, global_net, net_id, dataidxs, train_samples=1):
        losses = []
        net.to(args.device)
        global_net.to(args.device)
        train_dataloader, optimizer, criterion = train_handler(args, net, net_id, dataidxs)
        if args.optimizer == 'ivon': #IVON
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.01) #final learning rate is 0.01
            for epoch in range(args.epochs):
                for x, target in train_dataloader:
                    x, target = x.to(args.device), target.to(args.device)
                    for _ in range(train_samples):
                        with optimizer.sampled_params(train=True):
                            logit = net(x).to(args.device)
                            loss = criterion(logit, target)
                            loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    losses.append(loss.item())
        else: # classical variational inference
            for _ in range(args.epochs):
                for x, target in train_dataloader:
                    x, target = x.to(args.device), target.to(args.device)
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    out = net(x).to(args.device)
                    kl = net.KL() / net.dataset_size
                    loss = criterion(out, target) + kl
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())  
        #net.to("cpu")
        net.to(args.device)
        plot_losses( losses, net_id, args.round, args.logdir)
        torch.save(net.state_dict(), f"{args.logdir}/clients/client_{net_id}.pt")
        torch.save(optimizer.state_dict(), f"{args.logdir}/clients/client_{net_id}_optimizer.pt")

    def global_update(self, args, networks, selected, net_dataidx_map):
        fed_freqs = [1 / len(selected) for r in selected]
        update_global(args, networks, selected, net_dataidx_map, fed_freqs)

    def __str__(self):
        return "Bayesian Federated Learning algorithm"


class BFLAvg(BFL):
    def global_update(self, args, networks, selected, net_dataidx_map):
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        update_global(args, networks, selected, net_dataidx_map, fed_avg_freqs)
        
    def __str__(self):
        return "Bayesian Federated Learning algorithm with FedAVG global update rule"


class FED(BaseAlgorithm):
    def local_update(self, args, net, global_net, net_id, dataidxs):
        losses = []
        net.to(args.device)
        train_dataloader, optimizer, criterion = train_handler(args, net, net_id, dataidxs)
        for epoch in range(args.epochs):
            for x, target in train_dataloader:
                x, target = x.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        #net.to("cpu")
        net.to(args.device)
        plot_losses( losses, net_id, args.round, args.logdir)
        torch.save(net.state_dict(), f"{args.logdir}/clients/client_{net_id}.pt")
        torch.save(optimizer.state_dict(), f"{args.logdir}/clients/client_{net_id}_optimizer.pt")
        
    def global_update(self, args, networks, selected, net_dataidx_map):
        fed_freqs = [1 / len(selected) for r in selected]
        update_global(args, networks, selected, net_dataidx_map, fed_freqs)

    def __str__(self):
        return "Bayesian Federated Learning algorithm"


class FEDAvg(FED):     
    def global_update(self, args, networks, selected, net_dataidx_map):
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        update_global(args, networks, selected, net_dataidx_map, fed_avg_freqs)

    def __str__(self):
        return "Federated Learning algorithm with FedAVG global update rule"


class FEDProx(FEDAvg):
    def local_update(self, args, net, global_net, net_id, dataidxs):
        losses = []
        net.to(args.device)
        global_net.to(args.device)
        train_dataloader, optimizer, criterion = train_handler(args, net, net_id, dataidxs)
        global_weight_collector = list(global_net.to(args.device).parameters())
        for epoch in range(args.epochs):
            for x, target in train_dataloader:
                x, target = x.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss += fed_prox_reg
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        #net.to("cpu")
        #global_net.to("cpu")
        net.to(args.device)
        global_net.to(args.device)
        plot_losses(losses, net_id, args.round, args.logdir)
        torch.save(net.state_dict(), f"{args.logdir}/clients/client_{net_id}.pt")

    def __str__(self):
        return "Federated Learning algorithm with FedProx global update rule"


class FEDNova(BaseAlgorithm):
    def local_update(self, args, net, global_net, net_id, dataidxs):
        losses = []
        net.to(args.device)
        train_dataloader, optimizer, criterion = train_handler(args, net, net_id, dataidxs)
        for epoch in range(args.epochs):
            for x, target in train_dataloader:
                x, target = x.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        #net.to("cpu")
        net.to(args.device)
        tau = len(train_dataloader) * args.epochs
        a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
        global_net_para = global_net.state_dict()
        net_para = net.state_dict()
        norm_grad = deepcopy(global_net.state_dict())
        for key in norm_grad:
            norm_grad[key] = torch.true_divide(global_net_para[key]-net_para[key], a_i)
        torch.save(net.state_dict(), f"{args.logdir}/clients/client_{net_id}.pt")
        torch.save(norm_grad, f"{args.logdir}/clients/norm_grad_{net_id}.pt")
        plot_losses(losses, net_id, args.round, args.logdir)

    def global_update(self, args, networks, selected, net_dataidx_map):
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        norm_grad_total = deepcopy(networks["global_model"].state_dict())
        for key in norm_grad_total:
            norm_grad_total[key] = 0.0
        for i in enumerate(selected):
            norm_grad = torch.load(f"{args.logdir}/clients/norm_grad_{r}.pt")
            for key in norm_grad_total:
                norm_grad_total[key] += norm_grad[key] * freqs[i]
        coeff = 0.0
        for i, r in enumerate(selected):
            tau = math.ceil(len(net_dataidx_map[r])/args.batch_size) * args.epochs
            a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
            coeff = coeff + a_i * freqs[i]
        global_para = networks["global_model"].state_dict()
        for key in global_para:
            if global_para[key].type() == 'torch.LongTensor':
                global_para[key] -= (coeff * norm_grad_total[key]).type(torch.LongTensor)
            elif global_para[key].type() == 'torch.cuda.LongTensor':
                global_para[key] -= (coeff * norm_grad_total[key]).type(torch.cuda.LongTensor)
            else:
                global_para[key] -= coeff * norm_grad_total[key]
        networks["global_model"].load_state_dict(global_para)

    def __str__(self):
        return "Federated Learning algorithm with FedNOVA global update rule"


class Scaffold(BaseAlgorithm):
    def local_update(self, args, net, global_net, net_id, dataidxs):
        losses = []
        c_global_para = torch.load(f"{args.logdir}/clients/c_global.pt", map_location=args.device)
        c_local_para = torch.load(f"{args.logdir}/clients/c_{net_id}.pt", map_location=args.device)
        net.to(args.device)
        global_net.to(args.device)
        train_dataloader, optimizer, criterion = train_handler(args, net, net_id, dataidxs)
        cnt = 0
        for epoch in range(args.epochs):
            for x, target in train_dataloader:
                x, target = x.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)
                cnt += 1
                losses.append(loss.item())
        #net.to("cpu")
        net.to(args.device)
        c_new_para = torch.load(f"{args.logdir}/clients/c_{net_id}.pt")
        c_delta_para = torch.load(f"{args.logdir}/clients/c_{net_id}.pt")
        c_global_para = torch.load(f"{args.logdir}/clients/c_global.pt")
        c_local_para = torch.load(f"{args.logdir}/clients/c_{net_id}.pt")
        global_model_para = global_net.state_dict()
        net_para = net.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        torch.save(net.state_dict(), f"{args.logdir}/clients/client_{net_id}.pt")
        torch.save(c_new_para, f"{args.logdir}/clients/c_{net_id}.pt")
        torch.save(c_delta_para, f"{args.logdir}/clients/c_delta_{net_id}.pt")
        plot_losses(losses, net_id, args.round, args.logdir)

    def global_update(self, args, networks, selected, net_dataidx_map):
        total_delta = deepcopy(networks["global_model"].state_dict())
        for key in total_delta:
            total_delta[key] = 0.0
        for r in selected:
            c_delta_para = torch.load(f"{args.logdir}/clients/c_delta_{r}.pt")
            for key in total_delta:
                total_delta[key] += c_delta_para[key] / len(selected)
        c_global_para = torch.load(f"{args.logdir}/clients/c_global.pt")
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_para[key] += total_delta[key]
        torch.save(c_global_para, f"{args.logdir}/clients/c_global.pt")
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        global_para = networks["global_model"].state_dict()
        for i, r in enumerate(selected):
            #net_para = networks["nets"][r].cpu().state_dict()
            net_para = networks["nets"][r].to(args.device).state_dict()
            if i == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[i]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[i]
        networks["global_model"].load_state_dict(global_para)
    
    def __str__(self):
        return "Federated Learning algorithm with Scaffold global update rule"

