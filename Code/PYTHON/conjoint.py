import numpy as np
import pandas as pd
import utils
import time


def hbmnl(data_dict, mu=(0,1), alpha=(0,10), lkj_param=2, return_fit=False, **kwargs):
    """
    Hierarchical Bayesian Multi-Nomial Logit for conjoint analysis.

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
    nresp_train = data_dict['Xtrain'].shape[0]
    nresp_test = data_dict['Xtest'].shape[0]
    ntask_train = data_dict['Xtrain'].shape[1]
    ntask_test = data_dict['Xtest'].shape[1]
    N = nresp*ntask_train
    Ntest = nresp*ntask_test

    stan_data = {
            'A':nalts,
            'L':nlvls,
            'T':ntask_train,
            'R':nresp_train,
            'C':1,
            'Rtest':nresp_test,
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
    MODEL = utils.get_model(model_name='hbmnl')
    FIT = utils.fit_model_to_data(MODEL, stan_data, **kwargs)
    
    Yc = FIT.extract(pars=['Yc'])['Yc'].sum(axis=0).reshape((Ntest, nalts))
    Yhat = np.argmax(Yc, axis=1) + 1
    
    hit_count = Ntest - np.count_nonzero(Yhat - data_dict['Ytest'].reshape(Ntest))

    # store results
    results = dict()
    results["SCORE"] = hit_count/Ntest

    if return_fit:
        return results, FIT
    else:
        return results



### END ###
