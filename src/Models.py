import pyrcel as pm
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################
def setup_aerosols_distribution(param_vals, distribution_type='single'):

    #if param_vals[1].shape[0] == 1:
    if distribution_type is 'single':
        aerosol =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=0.015, sigma=1.6, N=param_vals[1]),kappa=0.15, bins=200)
    else:
        aerosol =  pm.AerosolSpecies('sulfate', pm.MultiModeLognorm(mus=0.015, sigmas=1.2, Ns=1000), kappa=0.54, bins=200)
    
    return aerosol

########################################################
def pyrcel_model(X):

    X_size = X.shape
    print('model input size:', X_size)
    print('X', X)

    onedim_output = []

    for pv in X:
        print('x1,x2,x3:',pv)
        #print(f'{pv:f}')
        
        # Input params
        V = pv[0]
        AC = pv[1]
        #KP = pv[2]

        # To do: a function that allows to switch between one mode and multi mode distributions.
        #aerso_distr = setup_aerosols_distribution(pv)
        aerosol_distribution =  pm.AerosolSpecies('sulfate', pm.Lognorm(mu=0.015, sigma=1.6, N=AC),kappa=0.15, bins=200)

        initial_aerosols = [aerosol_distribution]
        
        # Fixed
        P0 = 77500. # Pressure, Pa
        T0 = 274.   # Temperature, K
        S0 = -0.02  # Supersaturation, 1-RH (98% here)

        dt = 1.0 # timestep, seconds
        t_end = 250./V # end time, seconds... 250 meter simulation
        
        model = pm.ParcelModel(initial_aerosols, V, T0, S0, P0, console=False, accom=0.3)
        parcel_trace, aerosol_traces = model.run(t_end, dt, solver="cvode")
                
        s = parcel_trace['S'].to_numpy(dtype='float64')
        mu = np.mean(s)
        onedim_output.append(mu)
        #arr = np.expand_dims(s, axis=1)
        #print('onedim_output', arr.shape)
        #onedim_output.append(np.mean(arr))

    tmp = np.array(onedim_output)
    out = np.expand_dims(tmp, axis=1)
    print(out.shape)
        
    return out

########################################################
def model(X):
    """model _summary_

    _extended_summary_

    Parameters
    ----------
    X : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
        
    X_size = X.shape
    print('model input size:', X_size)
    print('X', X)

    onedim_output = []
    
    # Input params
    V = X[0][0]
    AC = X[0][1]
    #KP = X[0][2]

    '''
    ms = np.array([0.05, 0.045, 0.090])
    sgs = np.array([1.6, 2.3, 1.5])
    Nss = np.array([150, 450, 300])
    sulfate =  pm.AerosolSpecies('sulfate',
                            pm.MultiModeLognorm(mus=ms, sigmas=sgs, Ns=Nss),
                            kappa=0.54, bins=200)
    '''
        
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