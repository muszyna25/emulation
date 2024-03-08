import pyrcel as pm
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Util as ut

from pyrcel import binned_activation

class ParcelModel:

    def __init__(self):
        self.model = None 

    def build_parcel_model():
        '''
        inputs: 
            dict of inputs for Aerosol setup - mu or mus, etc

        return model init
        '''

# To Do: maybe create a class experiement to setup an experiement name and input data

########################################################
def construct_PPE_setup(setup, nsamples=5, ndim=2):

    #1. call function that generates new names
    '''
    param_values = []
    for k in list(setup.keys()):
        #s = np.array(list(setup[k]))
        s = setup[k]
        print(s)
        vals = ut.lhs_generate_input_data(s, nsamples, ndim)
        print(s, vals)
        param_values.append(vals)

    ppe_data = dict(map(lambda i,j : (i,j) , list(setup.keys()), param_values))
    '''

    datapoints = ut.lhs_generate_input_data(setup, nsamples, ndim)
    print('Initial parameter datapoints:', datapoints.shape, datapoints)

    return datapoints

########################################################
def setup_aerosols_distribution(params_vals, distribution_type='single'):

    #if param_vals[1].shape[0] == 1:
    aers = []
    if distribution_type is 'single':
        nsamples, nparams = params_vals.shape
        for i in range(0, nsamples):
            s = params_vals[i]
            aerosol =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=s[0], sigma=s[1], N=s[2]), kappa=s[3], bins=200)
            aers.append(aerosol)
    else:
        aerosol =  pm.AerosolSpecies('sulfate', pm.MultiModeLognorm(mus=0.015, sigmas=1.2, Ns=1000), kappa=0.54, bins=200)
    
    print(aers)

    return aers

########################################################
def parcel_model(X):

    X_size = X.shape
    print('model input size:', X_size)
    print('X', X)

    onedim_output = []
    
    # Input params
    V = X[0][0]
    AC = X[0][1]
    #KP = X[0][2]
        
    # Fixed
    P0 = 77500. # Pressure, Pa
    T0 = 274.   # Temperature, K
    S0 = -0.02  # Supersaturation, 1-RH (98% here)
    
    sulfate =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=0.015, sigma=1.6, N=AC),
                                kappa=0.15, bins=200)

    #sulfate =  pm.AerosolSpecies('sulfate',
    #                            pm.MultiModeLognorm(mus=0.015, sigmas=1.2, Ns=AC),
    #                            kappa=KP, bins=200)
    
    initial_aerosols = [sulfate]
        
    dt = 1.0 # timestep, seconds
    t_end = 250./V # end time, seconds... 250 meter simulation
        
    model = pm.ParcelModel(initial_aerosols, V, T0, S0, P0, console=False, accom=0.3)
    parcel_trace, aerosol_traces = model.run(t_end, dt, solver="cvode")
                
    s = parcel_trace['S'].to_numpy(dtype='float64')
    mu = np.mean(s)
    #onedim_output.append(mu)
    #arr = np.expand_dims(s, axis=1)
    #print('onedim_output', arr.shape)
    #onedim_output.append(np.mean(arr))
    
    tmp = np.array([mu])
    out = np.expand_dims(tmp, axis=1)
    print('model output size', out.shape)
        
    return out


########################################################
def pyrcel_model(X):

    X_size = X.shape
    print('model input size:', X_size)
    print('X', X)

    onedim_output = []
    smaxes = []

    for pv in X:
        print('xs:',pv)
        #print(f'{pv:f}')
        
        # Input params
        #V = pv[0]
        #AC = pv[1]
        #KP = pv[2]

        # To do: a function that allows to switch between one mode and multi mode distributions.
        #aerso_distr = setup_aerosols_distribution(pv)
        aerosol_distribution =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=pv[0], sigma=pv[1], N=pv[2]), kappa=pv[3], bins=200)

        initial_aerosols = [aerosol_distribution]
        
        # Fixed
        P0 = 77500. # Pressure, Pa
        T0 = 274.   # Temperature, K
        S0 = -0.02  # Supersaturation, 1-RH (98% here)
        V0 = 1.0

        dt = 1.0 # timestep, seconds
        t_end = 50./V0 # end time, seconds... 250 meter simulation
        
        model = pm.ParcelModel(initial_aerosols, V0, T0, S0, P0, console=False, accom=0.3)
        parcel_trace, aerosol_traces = model.run(t_end, dt=dt, solver="cvode", terminate=True)
        
        # first largest
        #print(parcel_trace['S'])
        ind = parcel_trace['S'].argmax() - 1
        parcel_trace['S'] = parcel_trace['S'].drop(ind)

        # second largest
        smax = parcel_trace['S'].max() * 100
        #time_at_smax = parcel_trace['S'].argmax()

        #wet_sizes_at_Smax = aerosol_traces['ammonium sulfate'].loc[time_at_smax].iloc[0]

        #wet_sizes_at_Smax = np.array(wet_sizes_at_Smax.tolist())

        #frac_eq, _, _, _ = binned_activation(smax, T0, wet_sizes_at_Smax, initial_aerosols)
        
        # Save the output
        smaxes.append(smax)
        #act_fracs.append(frac_eq)

        #s = parcel_trace['S'].to_numpy(dtype='float64')
        #mu = np.mean(s)
        #onedim_output.append(mu)

        #arr = np.expand_dims(s, axis=1)
        #print('onedim_output', arr.shape)
        #onedim_output.append(np.mean(arr))

    tmp = np.array(smaxes)
    out = np.expand_dims(tmp, axis=1)
    print(out.shape)
        
    return out

########################################################
def model(X):
        
    X_size = X.shape
    print('model input size:', X_size)
    print('X', X)

    onedim_output = []
    
    # Input params
    #V = X[0][0]
    #AC = X[0][1]
    #KP = X[0][2]
        
    # Fixed
    P0 = 77500. # Pressure, Pa
    T0 = 274.   # Temperature, K
    S0 = -0.02  # Supersaturation, 1-RH (98% here)
    V0 = 1.0
    
    aerosol =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=X[0][0], sigma=X[0][1], N=X[0][2]), kappa=X[0][3], bins=200)
    
    initial_aerosols = [aerosol]
        
    dt = 1.0 # timestep, seconds
    t_end = 50./V0# end time, seconds... 250 meter simulation
        
    model = pm.ParcelModel(initial_aerosols, V0, T0, S0, P0, console=False, accom=0.3)
    parcel_trace, aerosol_traces = model.run(t_end, dt=dt, solver="cvode", terminate=True)

    # first largest
    #print(parcel_trace['S'])
    ind = parcel_trace['S'].argmax() - 1
    parcel_trace['S'] = parcel_trace['S'].drop(ind)

    # second largest
    smax = parcel_trace['S'].max() * 100
    #time_at_smax = parcel_trace['S'].argmax()

    #wet_sizes_at_Smax = aerosol_traces['ammonium sulfate'].loc[time_at_smax].iloc[0]

    #wet_sizes_at_Smax = np.array(wet_sizes_at_Smax.tolist())

    #frac_eq, _, _, _ = binned_activation(smax, T0, wet_sizes_at_Smax, initial_aerosols)
                
    #s = parcel_trace['S'].to_numpy(dtype='float64')
    #mu = np.mean(s)
    
    #onedim_output.append(mu)
    #arr = np.expand_dims(s, axis=1)
    #print('onedim_output', arr.shape)
    #onedim_output.append(np.mean(arr))
    
    tmp = np.array([smax])
    out = np.expand_dims(tmp, axis=1)
    print('model output size', out.shape)
        
    return out

########################################################
def plot_inputs(Xt, amo, lims):
    """plot_inputs _summary_

    _extended_summary_

    Parameters
    ----------
    Xt : _type_
        _description_
    amo : _type_
        _description_
    lims : _type_
        _description_
    """

    dim = Xt.shape[1]

    # To do: Temporarly these labels are default.
    params = ['x','y','z']

    if dim < 3:  
        fig = plt.figure(figsize=[10,8])
        ax = fig.add_subplot(1, 3, 1)
        ax.scatter(amo.data['X'][:,0], amo.data['X'][:,1], alpha=0.2, c='green')
        ax.scatter(Xt[:,0], Xt[:,1], alpha=1, s=50, c='red')
        ax.set_ylabel(params[0]), ax.set_xlabel(params[1])
        ax.set_xlim(lims[0]), ax.set_ylim(lims[1])
        plt.show()
    else:
        fig = plt.figure(figsize=[10,8])
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter3D(amo.data['X'][:,0], amo.data['X'][:,1], amo.data['X'][:,2], marker='.', alpha=0.2, c='green', s=90)
        
        # Do I need this line?
        ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], marker='.', c='r', s=90)
        
        ax.set_xlim(lims[0]), ax.set_ylim(lims[1]), ax.set_zlim(lims[2])
        ax.set_title("Input data")
        ax.set_xlabel(params[0]), ax.set_ylabel(params[1]), ax.set_zlabel(params[2])
        plt.show()
    
def plot_fit(amo, lims):
    """plot_fit _summary_

    _extended_summary_

    Parameters
    ----------
    amo : _type_
        _description_
    lims : _type_
        _description_
    """

    # To do: Temporarly these labels are default.
    params = ['x','y','z']

    fig = plt.figure(figsize=[18,8])
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    
    M = 50; 
    grid0 = np.linspace(lims[0][0],lims[0][1],M).reshape(M,1); 
    grid1 = np.linspace(lims[1][0],lims[1][1],M).reshape(M,1);
    XX1,XX2 = np.meshgrid(grid0,grid1)
    
    # amo.emudict['emu0'] I will have many emulator if P >= 2 and need to change 'emuXXX'.
    Z = [[amo.emudict['emu0'].predict( np.array([XX1[i,j], XX2[i,j] ]).reshape(1,2) )[0,0] for i in range(M)] for j in range(M)]
    #Z = [[amo.emudict['emu0'].predict( np.array([XX1[i,j], XX2[i,j] ]).reshape(1,3) )[0,0] for i in range(M)] for j in range(M)]
    Z = np.array(Z).T
    
    #ax.plot_surface(XX1, XX2, Z, rstride=1, cstride=1, alpha=0.3, cmap='jet')
    ax.plot_surface(XX1, XX2, Z, rstride=1, cstride=1, alpha=0.3)
    #ax.scatter3D(XX1, XX2, Z, marker='.', alpha=0.4, c='r')
    ax.set_xlabel(params[0]), ax.set_ylabel(params[1]), ax.set_zlabel(params[2], labelpad=1.0)
    
########################################################
def plot_acq_func(Xt, Yt, amo, acq, lims, limitscale):
    """plot_acq_func _summary_

    _extended_summary_

    Parameters
    ----------
    Xt : _type_
        _description_
    Yt : _type_
        _description_
    amo : _type_
        _description_
    acq : _type_
        _description_
    lims : _type_
        _description_
    limitscale : _type_
        _description_
    """

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(1, 3, 3)
    
    M = 30
    lim0 = np.linspace(lims[0][0],lims[0][1],M)
    lim1 = np.linspace(lims[1][0],lims[1][1], M)
    x0, x1 = np.meshgrid(lim0, lim1)
    ack = [[ np.float64( acq(np.array([[ x0[i, j], x1[i, j] ]])).item() )
            for i in range(M)] for j in range(M)]
    ack = np.array(ack).T
    #print('ACK', ack)
    im=ax.imshow(ack, origin='lower',extent=[lims[0][0],lims[0][1],lims[1][0],lims[1][1]], interpolation='gaussian', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    #cax = fig.add_axes([0.67, 0.1, 0.5, 0.05])
    #fig.colorbar(im, cax=cax, orientation='vertical')
    
    ax.set_ylabel('x'), ax.set_xlabel('y') 
    ax.set_title("Acqusition function A(x)")
    
    # Update
    new = amo.update(limitscale, acq)
    print( amo.emudict['emu0'].likelihood_variance )
    ax.scatter(new[0],new[1],s=70,color='magenta',alpha=1)
    
    print('RMSE: ',amo.test(Xt,Yt))
    plt.show()