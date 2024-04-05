# Suppress warnings
import warnings
warnings.simplefilter('ignore')

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

import src.Models as mp
import src.Util as ut

from os.path import isfile

from src.modules.AMOGAPE import AMOGAPE

def amogape():
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
    acq = amo.A_add_D #amo.A_add_D_prod_G #amo.A_prod_D_prod_G

    iter_error = []
    limitscale = 1.0
    rate_means = []
    rate_stds = []
    #random_rmse = []
    for run in range(30):    
        print('\n ITERATION: ', run)
        # Update
        new = amo.update(limitscale, acq)
        #amo.force_likelihoodnoise(1e-9)

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
            #random_rmse.append(rmse1)
            '''
            lstdstmp = []
            for i in range(0, int(amo.data['X'].shape[1])):
                lstdstmp.append(np.std(amo.data['X'][:,i]))
            amo.stdevs = np.array(lstdstmp)
            '''
        print('Stdevs:', amo.stdevs)
        if not run%5:
            #amo.means = amo.means*np.exp(-0.5*1)
            amo.stdevs = amo.stdevs*np.exp(-0.1*1)
            rate_means.append(amo.means)
            rate_stds.append(amo.stdevs)
        
        mp.reduced_acq_func(Xt, amo, acq, ranges, run)

if __name__ == '__main__':

###########################################
#
#   Generate data
#
###########################################
    exp_id = str(1065)

    #param_names = ['mu', 'sigma', 'nc', 'kappa']
    #ranges = np.array([[0.02, 0.1], [1.0, 2.5], [800, 1000], [0.4, 0.7]])

    param_names = ['mu', 'sigma']
    #ranges = np.array([[0.05, 0.25], [0.9, 2.5]])
    ranges = np.array([[0.02, 0.07], [1.1, 2.0]])
    nsamples = 3

    exp_name = ut.get_simple_file_name(exp_id, nsamples)
    parent_dir = os.path.join(os.getcwd(), 'experiments')
    
    #if isfile('data/' + exp_name):
    exp_dir = os.path.join(parent_dir, exp_id)
    if os.path.isdir(exp_dir):
        print(exp_dir + '/' + exp_name)
        Xt, Yt = ut.load_data(exp_name, exp_dir)
        print('loaded data...', Xt.shape, Yt.shape)
    else:
        path = os.path.join(parent_dir, str(exp_id)) 
        new_dir = os.makedirs(path, exist_ok=True)

        Xt = ut.lhs_generate_input_data(ranges, nsamples, dim=int(len(param_names)))
        print(Xt)
        Yt = mp.pyrcel_model(Xt)
        #np.savez('data/' + exp_name, Xt, Yt)
        fn = os.path.join(exp_dir, exp_name)
        np.savez(fn, Xt, Yt)
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

    amo_ae = AMOGAPE(mp.model, D=D, P=P, means=means, stdevs=stdevs, inputlimits=ranges, startdata=startdata)
    #acq = amo_ae.A_add_D_prod_G
    #acq = amo_ae.A_add_D_add_G
    acq = amo_ae.A_add_D #amo.A_add_D_prod_G #amo.A_prod_D_prod_G

    amo_rnd_gauss = AMOGAPE(mp.model, D=D, P=P, means=means, stdevs=stdevs, inputlimits=ranges, startdata=startdata)

    amo_lhs = AMOGAPE(mp.model, D=D, P=P, means=means, stdevs=stdevs, inputlimits=ranges, startdata=startdata)
    
    ae_test = []
    rnd_gauss_test = []

    N_ITER = 30
    for run in range(N_ITER):    
        print('\n ITERATION: ', run)

        if not run%10:
            amo_ae.stdevs = amo_ae.stdevs*np.exp(-0.5*1)

        #if not run%10:
        if 1:
            # RMSE AE
            # error between fit Y' to training data Xt and the ground truth of parcel model Yt
            rmse_ae = amo_ae.test(Xt,Yt) 
            ae_test.append(rmse_ae)

            # RMSE RND
            rmse_rnd_gauss = amo_rnd_gauss.test(Xt,Yt)
            rnd_gauss_test.append(rmse_rnd_gauss)

        # Update AE and RND
        limitscale = 1.0
        amo_ae.update(limitscale, acq)
        amo_rnd_gauss.update_gauss()
        print('\n update lhs...\n')
        amo_lhs.update_LHS()
    
    np.save(exp_dir + '/' + 'rmse_ae_rnd.npy', ae_test, rnd_gauss_test)
    np.save(exp_dir + '/' + 'tracker_ae_rnd.npy', amo_ae.tracker, amo_rnd_gauss.tracker)

    #np.save(exp_dir + '/' + 'emulator_ae.npy', amo_ae.emudict['emu0'], allow_pickle=True)
    np.save(exp_dir + '/' + 'emulator_ae.npy', amo_ae, allow_pickle=True)
    np.save(exp_dir + '/' + 'emulator_rnd_gauss.npy', amo_rnd_gauss, allow_pickle=True)
    np.save(exp_dir + '/' + 'emulator_lhs.npy', amo_lhs, allow_pickle=True)

    '''
    #mp.plot_fit(amo_ae, ranges)
    #plt.show()
    #mp.plot_fit(amo_lhs, ranges)
    mp.plot_2d(amo_ae, ranges)
    plt.show()

    lims = ranges
    M = 30
    lim0 = np.linspace(lims[0][0],lims[0][1],M)
    lim1 = np.linspace(lims[1][0],lims[1][1], M)
    x0, x1 = np.meshgrid(lim0, lim1)
    ack = [[ np.float64(acq(np.array([[ x0[i, j], x1[i, j] ]])).item()) for i in range(M)] for j in range(M)]
    ack = np.array(ack).T

    #Implement saving for plotting results later
    #np.savez(exp_dir + '/' + 'rmse1', ae_test, lhs_test)
    np.save(exp_dir + '/' + 'ack.npy', ack)
    np.save(exp_dir + '/' + 'tracker.npy', amo_ae.tracker)
    '''
    
    #with open('tracker.pkl', 'wb') as f:
    #    pickle.dump(amo_ae.tracker, f)
    #with open('ae_data.pkl', 'wb') as f:
    #    pickle.dump(amo_ae.data, f)

    #plt.plot(np.linspace(0,N_ITER,3), ae_test,label='amogape')
    #plt.plot(np.linspace(0,N_ITER,3), lhs_test,label='LHS')
    #plt.legend()
    #plt.show()

    '''
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
    acq = amo.A_add_D #amo.A_add_D_prod_G #amo.A_prod_D_prod_G

    iter_error = []
    limitscale = 1.0
    rate_means = []
    rate_stds = []
    #random_rmse = []
    for run in range(30):    
        print('\n ITERATION: ', run)
        # Update
        new = amo.update(limitscale, acq)
        #amo.force_likelihoodnoise(1e-9)

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
            #random_rmse.append(rmse1)
            
            #lstdstmp = []
            #for i in range(0, int(amo.data['X'].shape[1])):
            #    lstdstmp.append(np.std(amo.data['X'][:,i]))
            #amo.stdevs = np.array(lstdstmp)
        print('Stdevs:', amo.stdevs)
        if not run%5:
            #amo.means = amo.means*np.exp(-0.5*1)
            amo.stdevs = amo.stdevs*np.exp(-0.1*1)
            rate_means.append(amo.means)
            rate_stds.append(amo.stdevs)
        
        mp.reduced_acq_func(Xt, amo, acq, ranges, run)
    '''  

    '''
    uncomment this block
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
    '''
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