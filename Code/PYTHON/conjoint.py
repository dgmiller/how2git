import numpy as np
import scipy as sp
import pandas as pd
import utils
import time
import ensemble

import matplotlib.pyplot as plt
from matplotlib import animation


def hbmnl(data_dict, mu=(0,1), alpha=(0,10), lkj_param=2, return_fit=False, model_name='hbmnl', **kwargs):
    """
    Hierarchical Bayesian Multi-Nomial Logit for discrete choice experiements.

    INPUT
        data_dict (dict)
        mu (tuple) mean and variance of the mean of the model prior
        alpha (tuple) mean and variance of the variance of the model prior
        lkj_param (float) adjusts the lkj covariance prior
        **kwargs

    RETURNS
        results (dict)

    """

    # define local variables
    nresp = data_dict['X'].shape[0]
    nalts = data_dict['X'].shape[2]
    nlvls = data_dict['X'].shape[3]
    ntask_train = data_dict['Xtrain'].shape[1]
    ntask_test = data_dict['Xtest'].shape[1]
    N = nresp*ntask_train
    Ntest = nresp*ntask_test

    stan_data = {
            'A':nalts,
            'L':nlvls,
            'T':ntask_train,
            'R':nresp,
            'C':1,
            'Ttest':ntask_test,
            'X':data_dict['Xtrain'],
            'Y':data_dict['Ytrain'].astype(np.int64),
            'Z':np.ones((nresp,1)),
            'Xtest': data_dict['Xtest'],
            'mu_mean': mu[0],
            'mu_scale': mu[1],
            'alpha_mean': alpha[0],
            'alpha_scale': alpha[1],
            'lkj_param': lkj_param
            }
    
    
    # fit model to data
    MODEL = utils.get_model(model_name=model_name)
    FIT = utils.fit_model_to_data(MODEL, stan_data, **kwargs)
    
    return FIT


def hbmnl_demo(nresp=2, ntask=10, nalts=2, nlvls=2, factorial=False, model='hbmnl'):
    """
    Demonstrates hbmnl on simulated data.
    RETURNS
        FIT (stan object) the fit stan model
        data_dict (dict) the simulated data in a dictionary
    """
    if factorial:
        data_dict = utils.generate_factorial_design(nresp, nalts, nlvls)
    else:
        data_dict = utils.generate_simulated_design(nresp, ntask, nalts, nlvls)
    data_dict = utils.compute_beta_response(data_dict)
    #data_dict = utils.partition_train_test(data_dict, holdout=2)
    
    data_dict['Xtrain'] = data_dict['X'].copy()
    data_dict['Ytrain'] = data_dict['Y'].copy()
    data_dict['Xtest'] = data_dict['X'].copy()
    data_dict['Ytest'] = data_dict['Y'].copy()
    
    FIT = hbmnl(data_dict,
                mu=[0,5],
                alpha=[0,10],
                lkj_param=5,
                return_fit=True,
                model_name=model,
                iter=2000,
                chains=4,
                control={'adapt_delta':.9},
                init_r=1)

    Yhat = FIT.extract(pars=['Yhat'])['Yhat']
    predictions = sp.stats.mode(Yhat)[0][0].astype(int)
    print(predictions-data_dict['Y'])
    num_errors = np.count_nonzero(predictions-data_dict['Y'])
    print("ACCURACY: ", (len(data_dict['Y'].flatten()) - num_errors)/len(data_dict['Y'].flatten()))

    return FIT, data_dict


def ensemble_demo(nresp, ntask, nalts, nlvls):
    data_dict = utils.generate_simulated_design(nresp, ntask, nalts, nlvls)
    data_dict = utils.compute_beta_response(data_dict)
    data_dict['Xtrain'] = data_dict['X'].copy()
    data_dict['Ytrain'] = data_dict['Y'].copy()
    data_dict['Xtest'] = data_dict['X'].copy()
    data_dict['Ytest'] = data_dict['Y'].copy()

    results = ensemble.ensemble(data_dict)
    return results


def plot01():
    F, data = hbmnl_demo(nresp=2, ntask=15, nalts=2, nlvls=3, model='hbmnl')
    print(F)
    B = F.extract(pars=['B'])['B']
    draws,nresp,nlvls = B.shape

    fig,ax = plt.subplots(nrows=nresp, ncols=nlvls, figsize=(8,8))
    for i in range(nresp):
        for j in range(nlvls):
            ax[i,j].hist(B[:,i,j], bins=75, color='grey', alpha=.8)
            ax[i,j].axvline(x=data['B'].T[i,j], color='red')

    plt.show()

    fig,ax = plt.subplots(nrows=1, ncols=nresp, figsize=(12,8))
    for i in range(nresp):
        ax[i].scatter(B[:,i,0], B[:,i,1], color='grey', alpha=.1)
        ax[i].scatter(data['B'].T[i,0], data['B'].T[i,1], marker='.', color='red')
        ax[i].axis('equal')
    plt.show()

    B = F.extract(pars=['B'])['B']

    fig, ax = plt.subplots(nrows=nresp, ncols=nlvls, sharex=True, sharey=True, figsize=(8,8))
    for i in range(nresp):
        for j in range(nlvls):
            ax[i,j].plot(B[:,i,j].flatten(),'o', markersize=2)

    plt.show()



if __name__ == "__main__":
    #plot01()
    print(ensemble_demo(2, 15, 2, 3))



### END ###
