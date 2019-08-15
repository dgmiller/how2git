import utils
import conjoint

def demo01(nresp, nalts, nlvls):
    data_dict = utils.generate_factorial_design(nresp, nalts, nlvls)
    data_dict = utils.compute_beta_response(data_dict)
    
    data_dict['Xtrain'] = data_dict['X'].copy()
    data_dict['Ytrain'] = data_dict['Y'].copy()
    data_dict['Xtest'] = data_dict['X'].copy()
    data_dict['Ytest'] = data_dict['Y'].copy()
    
    FIT = conjoint.hbmnl(
                data_dict,
                mu=[0,3.5],
                alpha=[0,1],
                return_fit=True,
                iter=1000,
                chains=4,
                init_r=1)
    return FIT




### END ###
