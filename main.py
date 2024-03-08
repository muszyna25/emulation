# Suppress warnings
import warnings
warnings.simplefilter('ignore')

import pyrcel as pm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import src.Models as mp
import src.Util as ut

from os.path import isfile

from src.modules.AMOGAPE import AMOGAPE

if __name__ == '__main__':

###########################################
#
#   Generate data
#
###########################################
    exp_id = 1030

    param_names = ['mu', 'sigma', 'nc', 'kappa']
    ranges = np.array([[0.015, 0.05], [1.0, 2.0], [800, 1000], [0.5, 0.8]])
    nsamples = 5

    exp_name = ut.get_simple_file_name(exp_id, nsamples)

    if isfile('data/' + exp_name):
        Xt, Yt = ut.load_data(exp_name)
        print('loaded data...', Xt.shape, Yt.shape)
    else:
        Xt = ut.lhs_generate_input_data(ranges, nsamples, dim=int(len(param_names)))
        print(Xt)
        Yt = mp.pyrcel_model(Xt)
        np.savez('data/' + exp_name, Xt, Yt)
        print('generated data...', Xt.shape, Yt.shape)

###########################################
#
#   Emulate
#
###########################################
    # Input dimension
    D=int(len(param_names))
    #Output dimension
    P=1

    print('Xt size:',Xt.shape)
    lmeans, lstds = [], []
    for i in range(0, int(Xt.shape[1])):
        lmeans.append(np.mean(Xt[:,i]))
        lstds.append(np.std(Xt[:,i]))

    means = np.array(lmeans)
    stdevs = np.array(lstds)
    print(means, stdevs)

    startdata={}
    startdata['X']=Xt
    startdata['Y']=Yt #*1000 # I need to scale this and the output of model(x), but I forgot to do that.

    amo = AMOGAPE(mp.model, D=D, P=P, means=means, stdevs=stdevs, inputlimits=ranges, startdata=startdata)
    acq = amo.A_add_D_prod_G #amo.A_prod_D_prod_G

    iter_error = []
    limitscale = 1.0
    rate_means = []
    rate_stds = []
    for run in range(40):    
        # Update
        new = amo.update(limitscale, acq)
        amo.force_likelihoodnoise(1e-9)

        #print(amo.data['X'][:,0].shape, Xt[:,0].shape)
        #emup = amo.emudict['emu0']
        #Yp = emup.predict(amo.data['X'][:,0]).ravel()
        #print(Yp, Yt)

        #plt.figure()
        #plt.scatter(Xt[:,0],Yt, c='b', alpha=0.2)
        #plt.scatter(amo.data['X'][:,0],Yp, c='r', alpha=0.5)
        #plt.show()

        #print(amo.emudict['emu0'].likelihood_variance)
        rmse = amo.test(Xt,Yt)
        print('RMSE: ', rmse)
        if not run%10:
            iter_error.append(rmse)
            amo.means = amo.means*np.exp(-0.25*1)
            amo.stdevs = amo.stdevs*np.exp(-0.25*1)
            rate_means.append(amo.means)
            rate_stds.append(amo.stdevs)
            '''
            lstdstmp = []
            for i in range(0, int(amo.data['X'].shape[1])):
                lstdstmp.append(np.std(amo.data['X'][:,i]))
            amo.stdevs = np.array(lstdstmp)
            '''
    
    np.savez('data/' + str(exp_id) +'_rmse', iter_error)
    np.savez('data/' + str(exp_id) +'_means_stds', rate_means, rate_stds)

    #plt.figure()
    #plt.semilogy(list(range(0, int(len(iter_error)))), iter_error)
    #plt.show()

    #lpred = []
    Xtest = ut.lhs_generate_input_data(ranges, nsamples, dim=int(len(param_names)))
    emup = amo.emudict['emu0']
    Yp = emup.predict(Xtest).ravel()
    print('prediction:', Yp)
    Ym = mp.pyrcel_model(Xtest)

    np.savez('data/' + str(exp_id) + '_prediction_physmodel_vs_ml', Xtest, Yp, Ym)

    #plt.figure()
    #plt.scatter(Xtest[:,1],Ym, c='b', alpha=0.5)
    #plt.scatter(Xtest[:,1],Yp, c='r', alpha=0.5)
    #plt.show()

    #for xtest in range(0,Xtest.shape[0]):
    #    emup = amo.emudict['emu0']
        #Yp = emup.predict(np.array([[0.015], [1.0], [1000], [0.8]])).ravel()
    #    Yp = emup.predict(xtest).ravel()
    #    print('prediction:', Yp)

    #emu_p = .emudict['emu'+str(p)]
    #Yp = emu_p.predict(Xt).ravel()

    #mp.plot_inputs(Xt, amo, ranges)
    #mp.plot_fit(amo, ranges)
    #mp.plot_acq_func(Xt, Yt, amo, acq, ranges, limitscale)

    '''
        mu, sigma, number concentration, kappa, vdraft, 
        any other I wish to change? so it must be flexible
    '''

    '''
    #param_names = ['mu', 'sigma', 'nc', 'kappa', 'vdraft']
    #ranges = np.array([[0.5, 0.9], [0.05, 0.1], [80, 250], [0.1, 0.4], [0.1, 1.0]])
    
    param_names = ['mu', 'sigma', 'nc', 'kappa']
    ranges = np.array([[0.5, 0.9], [0.05, 0.1], [80, 250], [0.1, 0.4]])
    nsamples = 10

    # (start, stop, step)
    # What I am trying to vary? Is it same as for the example with vupdraft??
    #setup = dict(map(lambda i,j : (i,j) , param_names, ranges))
    #print(setup)

    params_values = mp.construct_PPE_setup(ranges, nsamples, ndim=int(len(param_names)))
    aer = mp.setup_aerosols_distribution(params_values, distribution_type='single')

    for i in range(0, nsamples):
        y_data = mp.parcel_model(aer)
    '''

    '''
    s = np.array([[1,10],[800,1000], [0.20, 0.90]]) # 3 params
    a = ut.lhs_generate_input_data(s, 10, 3) # this will genertte (10,3) matrix
    print(a.shape, a)
    '''