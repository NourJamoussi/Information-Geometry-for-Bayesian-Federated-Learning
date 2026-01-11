import numpy as np
import torch 
from collections import defaultdict
from .aggregation_functions import eaa, gaa, aalv, \
     forward_kl_barycenter, reverse_kl_barycenter , wasserstein_barycenter_diag

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
        #print(cov_agg[k].shape)
        #print(shape)
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

########################### Personalization Function ########################### 

#for ivon: 
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


def personalized_global(args, networks, net_id, personalization_weight): 
    assert args.arch in [ "bcnn" , "bcnn_1", "bcnn_2" , "bfcnn", "bcnn_speech"] or args.optimizer == 'ivon' , f"The architecture is not bayesian, personalization is not applied"
    method=args.perso_method #method of personalization (and so of the projection)
    assert method in ['wb_diag', 'rkl'] , f"The update method {method} is not barycentric, personalization is not applied"
    server=networks["global_model"] # the global model
    nets_to_average = [networks["nets"][net_id], server] #selected client model and the global model
    weights = [personalization_weight, 1-personalization_weight] #personalization weight corresponds to how much we want to be close to the local = lambda / (lambda +1)
    personalized_dict = aggregate(server, nets_to_average, method, weights)
    return personalized_dict




def personalized_global_ivon(args, networks, net_id, optimizers, personalization_weight): 
    method = args.perso_method #method of personalization (and so of the projection)
    assert method in ['wb_diag', 'rkl'] , f"The update method {method} is not barycentric, personalization is not applied"
    weights = [personalization_weight, 1-personalization_weight] #personalization weight corresponds to how much we want to be close to the local = lambda / (lambda +1)
    # Load the local optimizer and get the covariance
    loc_optim = optimizers["local_optimizers"][net_id]
    loc_ess = loc_optim.state_dict()['param_groups'][0]['ess']
    loc_hess = loc_optim.state_dict()['param_groups'][0]['hess']
    loc_wd = loc_optim.state_dict()['param_groups'][0]['weight_decay']
    loc_cov =  1 / (loc_ess * loc_hess + loc_wd)
    # Load the global optimizer
    global_optim = optimizers["global_optimizer"]
    # Load the local model 
    local_model = networks["nets"][net_id]
    # Load the global model
    global_model = networks["global_model"]
    # Personalize the local model
    means = [flatten_model_state_dict(local_model.state_dict()) , flatten_model_state_dict(global_model.state_dict())]
    covs = [loc_cov,global_optim.state_dict()['param_groups'][0]['cov']]
    means = torch.stack(means)
    covs = [torch.from_numpy(cov) if isinstance(cov, np.ndarray) else cov for cov in covs]
    covs = [cov.squeeze() if cov.dim() == 2 and cov.shape[1] == 1 else cov for cov in covs]
    covs = [cov.view(-1) for cov in covs]  # ensure 1D
    covs = torch.stack(covs)
    mu_agg , cov_agg = aggregate_param(means, covs, args.perso_method, weights)
    projected_net_dict = unflatten_model_state_dict(mu_agg, local_model.state_dict())
    return projected_net_dict, cov_agg