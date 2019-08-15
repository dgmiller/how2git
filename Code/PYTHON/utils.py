import pystan
import pickle
import numpy as np
import pandas as pd
from itertools import product, combinations


def generate_factorial_design(nresp, nalt, nlvls, ncovs=1):
    """
    Generates a factorial design.
    """
    P = []
    C = []
    for p in product([0,1], repeat=nlvls):
        P.append(list(p))

    for c in combinations(P, nalt):
        C.append(list(c))

    C = np.array(C)
    X = np.zeros((nresp, C.shape[0], nalt, nlvls))
    for r in range(nresp):
        X[r,:,:,:] = C
        Z = np.ones((nresp,1))

    data_dict = {'X':X,
                 'Z':Z,
                 'A':nalt,
                 'R':nresp,
                 'C':ncovs,
                 'T':C.shape[0],
                 'L':nlvls}

    return data_dict


def generate_simulated_design(nresp, ntask, nalts, nlvls, ncovs=1):
    """
    Generates a simulated design
    """
    # X is the experimental design
    X = np.zeros((nresp, ntask, nalts, nlvls))
    # Z is a matrix for demographic attributes
    Z = np.zeros((ncovs, nresp))
    
    for resp in range(nresp):
        z_resp = 1
        if ncovs > 1:
            raise NotImplementedError
    
        for scn in range(ntask):
            X_scn = np.random.uniform(0,1,size=nalts*nlvls).reshape(nalts, nlvls)
            #X_scn = np.random.choice([0,1], p =[.5,.5], size=nalts*nlvls).reshape(nalts,nlvls)
            X[resp, scn] += X_scn
    
        Z[:, resp] += z_resp
    
    # dictionary to store the simulated data and generation parameters
    data_dict = {'X':X,
                 'Z':Z.T,
                 'A':nalts,
                 'R':nresp,
                 'C':ncovs,
                 'T':ntask,
                 'L':nlvls}
    return data_dict


def compute_beta_response(data_dict, add_noise=True):
    """
    Computes the parameters Beta and response Y given design X.
    """

    # beta means
    Gamma = np.random.uniform(-3,4,size=data_dict['C'] * data_dict['L'])

    # beta variance-covariance
    Vbeta = np.diag(np.ones(data_dict['L'])) + .5 * np.ones((data_dict['L'], data_dict['L']))

    # Y is the response
    Y = np.zeros((data_dict['R'], data_dict['T']))
    # Beta is the respondent coefficients (part-worths/utilities)
    Beta = np.zeros((data_dict['L'], data_dict['R']))
    
    for resp in range(data_dict['R']):
        z_resp = 1
        if data_dict['C'] > 1:
            raise NotImplementedError
    
        beta = np.random.multivariate_normal(Gamma, Vbeta)
    
        for scn in range(data_dict['T']):
            X_scn = data_dict['X'][resp, scn]

            U_scn = X_scn.dot(beta)
            if add_noise:
                U_scn -= np.log(-np.log(np.random.uniform(size=data_dict['C'])))

            Y[resp, scn] += np.argmax(U_scn) + 1
    
        Beta[:, resp] += beta

    data_dict['B'] = Beta
    data_dict['Y'] = Y.astype(int)

    return data_dict



def get_model(model_name='hbmnl'):

    with open('../STAN/{0}.stan'.format(model_name), 'r') as f:
        stan_model = f.read()
    
    try:
        sm = pickle.load(open('../STAN/{0}.pkl'.format(model_name), 'rb'))
    
    except:
        sm = pystan.StanModel(model_code=stan_model)
        with open('../STAN/{0}.pkl'.format(model_name), 'wb') as f:
            pickle.dump(sm, f)
    
    return sm


def fit_model_to_data(model, data, **kwargs):
    """Runs the Stan sampler for model on data according to kwargs."""
    return model.sampling(data, **kwargs)


def partition_train_test(data_dict, holdout=5):
    data_dict['Xtrain'] = data_dict['X'][:,:-holdout,:,:].copy()
    data_dict['Ytrain'] = data_dict['Y'][:,:-holdout].copy()
    data_dict['Xtest'] = data_dict['X'][:,-holdout:,:,:].copy()
    data_dict['Ytest'] = data_dict['Y'][:,-holdout:].copy()
    return data_dict



### END ###
