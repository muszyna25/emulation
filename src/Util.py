
from pyDOE import lhs
import numpy as np
import datetime
import os
import random

########################################################
def lhs_generate_input_data(lims, n, dim=2):
    """lhs_generate_input_data _summary_

    _extended_summary_

    Parameters
    ----------
    lims : _type_
        _description_
    n : _type_
        _description_
    dim : int, optional
        _description_, by default 2

    Returns
    -------
    _type_
        _description_
    """

    diffs = np.diff(lims)
    Xt = lhs(dim, n)*diffs.T + lims[:,0].T 

    # self.data['X'] = lhs(self.D, self.N)*self.limdiffs.T + self.inputlimits[:,0].T

    return Xt

########################################################
def load_data(name, path='data/'):
    """load_data _summary_

    _extended_summary_

    Parameters
    ----------
    name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    npz = np.load(path + '/' + name)

    Xt = npz['arr_0']
    Yt = npz['arr_1']

    return Xt,Yt

########################################################
def generate_file_name(args, HEAD):

    date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    seedstr = str(args.seed).zfill(3)
    suffix = "{}_{}_{}_{}".format(args.alg, date, seedstr)
    result_name = os.path.join(HEAD, suffix)

    return result_name

########################################################
def get_simple_file_name(exp_id, n_samples):
    exp_setup = ['exp', str(exp_id), 'XY', str(n_samples), 'data', '.npz']
    exp_name = '_'.join(exp_setup)
    print(exp_name)
    return exp_name