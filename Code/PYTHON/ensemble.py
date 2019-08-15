import numpy as np
import pandas as pd
import utils
import time



def ensemble(data_dict, base_model='base_mnl', meta_model='meta_mnl', **kwargs):
    """
    Stacking Ensemble for conjoint analysis.

    INPUT
        data_dict (dict)
        **kwargs

    RETURNS
        results (dict)

    """

    # define local variables
    nresp = data_dict['X'].shape[0]
    ntask = data_dict['X'].shape[1]
    nalts = data_dict['X'].shape[2]
    nlvls = data_dict['X'].shape[3]
    ntask_train = data_dict['Xtrain'].shape[1]
    ntask_test = data_dict['Xtest'].shape[1]
    N = nresp*ntask_train
    Ntest = nresp*ntask_test
    K = 2
    M = nlvls
    
    # initialize and format data for stan model
    data = {}
    data['Xtrain'] = data_dict['Xtrain'].copy()
    data['Xtest'] = data_dict['Xtest'].copy()
    data['Ytrain'] = data_dict['Ytrain'].astype(np.int64)
    data['Ytest'] = data_dict['Ytest'].astype(np.int64)
    stan_data = {
            'A':nalts,
            'L':nlvls-1,
            'R':nresp,
            'Rtest':nresp,
            'loc':0,
            'scale':2
            }
    
    # fit model to data
    Tstep = [0, ntask_train//2, ntask_train]

    Yhat_train = np.zeros((N, nalts, M))
    for k in range(K):
        for m in range(M):
    
            Tk_fold = np.array([True]*ntask_train)
            Tk_fold[Tstep[k]:Tstep[k+1]] = False
            np.random.shuffle(Tk_fold)
   
            # set the K-fold temporary values of N and Ntest
            stan_data['T'] = sum(Tk_fold)
            stan_data['Ttest'] = sum(~Tk_fold)
    
            # new training set = subset of full training set
            # LOVO = Leave One Variable Out
            Xtrain_lovo = np.delete(data['Xtrain'][:, Tk_fold, :, :], m, 3)
            Xtest_lovo = np.delete(data['Xtrain'][:, ~Tk_fold, :, :], m, 3)
    
            stan_data['X'] = Xtrain_lovo
            stan_data['Y'] = data['Ytrain'][:, Tk_fold]
        
            # new test set = complement of new training set | full training set
            stan_data['Xtest'] = Xtest_lovo
        
            MODEL = utils.get_model(model_name=base_model)
            FIT = utils.fit_model_to_data(MODEL, stan_data, **kwargs)
        
            Yc = FIT.extract(pars=['Yc'])['Yc'].sum(axis=0).reshape((nresp*stan_data['Ttest'], nalts))
            Yhat_k = np.argmax(Yc, axis=1)

            kfold = np.array([False]*N).reshape(nresp, ntask_train)
            kfold[:, ~Tk_fold] = True
            kfold = kfold.flatten()
            Yhat_train[kfold, Yhat_k, m] += 1

    # make predictions on full test set using full training set
    model_scores = []
    Yhat_test = np.zeros((Ntest, nalts, M))
    for m in range(M):
    
        stan_data['T'] = ntask_train
        stan_data['Ttest'] = ntask_test
        
        # full training set
        Xtrain_lovo = np.delete(data['Xtrain'], m, 3)
        Xtest_lovo = np.delete(data['Xtest'], m, 3)
    
        stan_data['X'] = Xtrain_lovo
        stan_data['Y'] = data['Ytrain']
     
        # full test set
        stan_data['Xtest'] = Xtest_lovo
    
        MODEL = utils.get_model(model_name=base_model)
        FIT = utils.fit_model_to_data(MODEL, stan_data, **kwargs)
    
        Yc_test = FIT.extract(pars=['Yc'])['Yc'].sum(axis=0).reshape(Ntest, nalts)
        Yhat_k = np.argmax(Yc_test, axis=1)
        Yhat_test[np.array([True]*Ntest), Yhat_k, m] += 1
    
        model_scores.append(Ntest - np.count_nonzero(Yhat_k+1 - data['Ytest'].flatten()))
    
    
    # Fit stacking model to full test data using augmented training set
    stan_data['N'] = nresp*ntask_train
    stan_data['Ntest'] = Ntest
    stan_data['M'] = M
    stan_data['Yhat_train'] = Yhat_train.copy()
    stan_data['Yhat_test'] = Yhat_test.copy()
    stan_data['L'] = nlvls
    stan_data['Y'] = data['Ytrain'].flatten()
    
    MODEL = utils.get_model(model_name=meta_model)
    FIT = utils.fit_model_to_data(MODEL, stan_data, **kwargs)
    
    Yc_stacking = FIT.extract(pars=['Yc'])['Yc'].sum(axis=0).reshape((Ntest, nalts))
    Yhat_stacking = np.argmax(Yc_stacking, axis=1) + 1
    model_weights = FIT.extract(pars=['B'])['B']
    
    ensemble_hit_count = Ntest - np.count_nonzero(Yhat_stacking - data['Ytest'].flatten())
    
    # store results
    results = dict()
    results["SCORE"] = ensemble_hit_count/Ntest
    results["BASE MODEL SCORES"] = np.array(model_scores)/Ntest
    results["MODEL WEIGHTS"] = np.around(model_weights.mean(axis=0),decimals=2)
    
    yy = Yhat_test.sum(axis=2)
    
    coverage_list = []
    for j in range(Ntest):
        coverage_list.append(max(yy[j, :]))
    coverage = np.array(coverage_list)
    
    model_coverage = np.zeros((M,2))
    for i in range(M):
        model_coverage[i,0] = i+1
        model_coverage[i,1] = len(coverage[coverage==i+1])
    results["BASE MODEL COVERAGE"] = model_coverage

    return results




## END ##
