import numpy as np
import torch 
import ivon
from collections import defaultdict
from utils.FL.aggregation_functions import eaa, gaa, aalv, \
     forward_kl_barycenter, reverse_kl_barycenter , wasserstein_barycenter_diag
from utils.ivon_utils import IVON_SAMP
from utils.FL.utils import train_handler

########################### Aggregation Utils for Bayesian NN ###########################

def arrange_parameters(clients): 
    mu_dict = defaultdict(list)
    logsig2_dict = defaultdict(list)
    delta_dict = defaultdict(list)
    for client in clients:
        for key in client.state_dict().keys():
            if 'mu' in key and 'prior' not in key:
                mu_dict[key].append( client.state_dict()[key])
            elif 'sig' in key and 'prior' not in key: #rho is sigma 
                logsig2_dict[key].append( client.state_dict()[key])
            elif 'prior' not in key: 
                delta_dict[key].append( client.state_dict()[key])
    return mu_dict, logsig2_dict , delta_dict

def means_param(mu_dict, key):
    means = []
    for element in mu_dict[key]: 
        means.append(element.flatten())
    return np.array(means)


def covs_param(sigma_dict, key):
    covs = []
    for element in sigma_dict[key]: 
        diag_cov = np.transpose(np.exp(element.flatten()))
        #print(diag_cov.shape)
        covs.append(diag_cov)
    return np.array(covs) # list of the diagonal of the covariance matrices and not the covariances themselves


def delta_param(delta_dict, key):
    delta = []
    for element in delta_dict[key]: 
        delta.append(element.flatten())
    return np.array(delta)

def aggregate_param(means, covs, method, weights=None): 
    if method == 'eaa': 
        mu , cov = eaa(means, covs, weights) # works with the diagonals of the covariance matrices
        return mu, cov
    elif method == 'wb_diag': 
        mu , cov = wasserstein_barycenter_diag(means, covs, weights)
        return mu, cov
    elif method == 'fkl': 
        mu , cov = forward_kl_barycenter(means, covs, weights)
        return mu, cov
    elif method == 'rkl': 
        mu , cov = reverse_kl_barycenter(means, covs, weights)
        return mu, cov
    elif method == 'gaa':
        mu , cov = gaa(means, covs, weights) 
        return mu, cov
    elif method == 'aalv': 
        mu , cov = aalv(means, covs, weights) 
        return mu, cov
    else: 
        raise ValueError(f"update method {method} non implemented!")


def aggregate_delta(delta, method, weights=None): 
    if method in ['eaa', 'wb_diag', 'fkl', 'rkl', 'gaa', 'aalv']: 
        delta_avg = np.average(delta, weights=weights, axis=0)
        return delta_avg
    else: 
        raise ValueError(f"update method {method} non implemented!")


def aggregate_params(clients, method, weights=None): 
    mu_agg= defaultdict(list)
    cov_agg= defaultdict(list)
    delta_agg= defaultdict(list)
    mu_dict, logsig2_dict , delta_dict = arrange_parameters(clients)
    for key_mu , key_sigma in zip(mu_dict.keys(), logsig2_dict.keys()): 
        means = means_param(mu_dict, key_mu)
        covs = covs_param(logsig2_dict, key_sigma)  
        mu_key_agg , cov_key_agg = aggregate_param(means, covs, method, weights) 
        mu_agg[key_mu] = mu_key_agg
        cov_agg[key_sigma] = cov_key_agg
    
    for key_delta in delta_dict.keys():
        delta = delta_param(delta_dict, key_delta)
        delta_agg[key_delta] = aggregate_delta(delta, method, weights)
    
    return mu_agg, cov_agg , delta_agg


def detransform_params(mu_agg, cov_agg, delta_agg, shapes): 
    state_dict = {}
    for k in mu_agg.keys(): 
        shape = shapes[k]
        state_dict[k] = mu_agg[k].reshape(shape)
    for k in cov_agg.keys(): 
        shape = shapes[k]
        #diag = np.diag(cov_agg[k])
        sigma2 = cov_agg[k].reshape(shape)
        log_sigma2 = np.log(sigma2) 
        state_dict[k] = log_sigma2.reshape(shape)
    
    for k in delta_agg.keys(): 
        shape = shapes[k]
        state_dict[k] = delta_agg[k].reshape(shape)
    return state_dict

def aggregate(server, clients, method, weights=None): 
    shapes = {k: server.state_dict()[k].shape for k in server.state_dict().keys()}
    mu_agg , cov_agg, delta_agg = aggregate_params(clients, method, weights) 
    agg_state_dict = detransform_params(mu_agg, cov_agg, delta_agg, shapes)
    for key in agg_state_dict.keys():
        agg_state_dict[key] = torch.from_numpy(agg_state_dict[key])
    for key in server.state_dict().keys(): 
        if key not in agg_state_dict.keys(): 
            agg_state_dict[key] = server.state_dict()[key]
    agg_state_dict_ordered = {k: agg_state_dict[k] for k in server.state_dict().keys() if k in agg_state_dict}
    return agg_state_dict_ordered

########################### Aggregation Utils for IVON ###########################

def flatten_model_state_dict(state_dict):
    vec = []
    for param_tensor in state_dict:
        vec.append(state_dict[param_tensor].view(-1))
    return torch.cat(vec)

def unflatten_model_state_dict(vec, state_dict):
    state_dict = state_dict.copy()
    idx = 0
    for param_tensor in state_dict:
        param = state_dict[param_tensor]
        size = param.numel()
        state_dict[param_tensor] = torch.from_numpy( vec[idx:idx+size].reshape(param.size()) )
        idx += size
    return state_dict

def aggregate_ivon(args, networks, selected, net_dataidx_map, weights=None):
    means = []
    covs = []
    for net_id in selected:
        means.append(flatten_model_state_dict(networks["nets"][net_id].state_dict()))
        dataidxs = net_dataidx_map[net_id]
        train_dataloader, _, _ = train_handler(args, networks["nets"][net_id], net_id, dataidxs)
        N = len(train_dataloader.dataset) 
        optimizer = ivon.IVON(networks["nets"][net_id].parameters(), lr=args.lr, ess=N, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init)
        optimizer.load_state_dict(torch.load(f"{args.logdir}/clients/client_{net_id}_optimizer.pt"))        
        cov = 1 / (N * optimizer.state_dict()['param_groups'][0]['hess'] + optimizer.state_dict()['param_groups'][0]['weight_decay'])
        covs.append(cov)
    means = torch.stack(means)
    covs = torch.stack(covs)
    mu_agg , cov_agg = aggregate_param(means, covs, args.update_method, weights)
    agg_state_dict = unflatten_model_state_dict(mu_agg, networks["global_model"].state_dict())
    return agg_state_dict , cov_agg 

    

########################### Update Function ###########################
def update_global(args, networks, selected, net_dataidx_map, freqs): 
    
    print(f"updating global model after round {args.round}")

    if args.optimizer == 'ivon': 
        agg_state_dict , cov_agg  = aggregate_ivon(args, networks, selected, net_dataidx_map, freqs)
        networks["global_model"].load_state_dict(agg_state_dict)
        optimizer = IVON_SAMP(networks["global_model"].parameters(), lr=args.lr, ess=1, weight_decay=args.reg, beta1=args.rho, hess_init=args.hess_init, cov=cov_agg) #the ess is set to be 1 to pass the assertion. In all the cases it won't be used 
        torch.save(optimizer.state_dict(), f"{args.logdir}/global_optimizer.pt")

    else : 
        if args.arch in ["cnn", "fcnn", "cnn_speech"]:  #Fed
            global_para = networks["global_model"].state_dict()
            for idx, net_id  in enumerate(selected):
                net_para = networks["nets"][net_id].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * freqs[idx]
            networks["global_model"].load_state_dict(global_para)

        elif args.arch in [ "bcnn" , "bcnn_1", "bcnn_2" , "bfcnn", "bcnn_speech"]:  #BFL
            server=networks["global_model"] # the global model
            clients=[networks["nets"][net_id] for net_id in selected] #selected client models
            method=args.update_method #method of aggregation
            weights=freqs
            agg_state_dict = aggregate(server, clients, method, weights)
            networks["global_model"].load_state_dict(agg_state_dict)
        
        else:
            raise ValueError("Wrong arch!")

