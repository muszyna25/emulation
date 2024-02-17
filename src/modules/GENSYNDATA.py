import autograd.numpy as np
from scipy.stats import multivariate_normal

def model(X):
    return np.reshape( ( 2*X[:,0]**2 + X[:,1]**2 )* np.sin(X[:,1]/3), (np.shape(X)[0], 1) )

def trunc_gauss_sample(N,means,stdevs,lims):
    data = np.zeros([N,len(means)])
    n = 0
    nor2d = multivariate_normal(means, np.diag(np.square(stdevs)) )
    while N > n:
        point = nor2d.rvs()
        if (lims[0,0] < point[0] < lims[0,1]) & (lims[1,0] < point[1] < lims[1,1]):
            data[n,:] = point
            n += 1
    return(data)
