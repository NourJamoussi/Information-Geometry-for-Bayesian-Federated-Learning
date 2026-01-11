import numpy as np 


def eaa(means, covs, weights=None): #Empirical Arithmetic Aggregation
    mu = np.average(means.cpu(), weights=weights, axis=0)
    cov = np.average(covs.cpu(), weights=weights, axis=0)
    return mu, cov   

def gaa(means, covs, weights=None): #Gaussian Arithmetic Aggregation
    mu = np.average(means.cpu(), weights=weights, axis=0)
    weights_squared = [weight**2 for weight in weights]
    cov = np.average(covs.cpu(), weights=weights_squared, axis=0)  
    return mu, cov 

def aalv(means, covs, weights=None): #Arithmetic Aggregation with Log Variance 
    mu = np.average(means.cpu(), weights=weights, axis=0)
    cov = np.exp(np.average(np.log(covs.cpu()), weights=weights, axis=0))
    return mu, cov


def forward_kl_barycenter(means, covs, weights=None): 
    mu = np.average(means, weights=weights, axis=0)
    toavg = covs + (means - mu)**2
    cov = np.average(toavg, weights=weights, axis=0)
    return mu, cov 

def reverse_kl_barycenter(means, covs, weights=None):  # Kullback Leiber Average 
    means = means.cpu()
    covs = covs.cpu()
    inverted_covs = [1/cov for cov in covs]
    cov = 1/(np.average(inverted_covs, weights=weights, axis=0))
    inverted_covs_time_means = [(inverted_covs[i] * means[i]).reshape(-1,1) for i in range(len(means))]
    cov = cov.reshape(-1,1)
    mu = cov * np.average(inverted_covs_time_means, weights=weights, axis=0)
    return mu, cov


def wasserstein_barycenter_diag(means, covs, weights = None):
    #check if matrices are diagonal
    #assert all([ np.sum(K > 1e-10) == K.shape[0] for K in covs]), \
    #    "NotDiagonal: One of the covariance matrices is not diagonal."
    mu = np.average(means.cpu(), weights=weights, axis=0)
    cov = np.average([np.sqrt(K.cpu()) for K in covs], weights=weights, axis=0)**2  # for ivon 
    #cov = np.average([np.sqrt(K) for K in covs], weights=weights, axis=0)**2  #for non ivon 
    return mu, cov

