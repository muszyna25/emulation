import autograd
import autograd.numpy as np
from sklearn.decomposition import PCA
from pyDOE import lhs
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from src.mintGP import mintGP
from scipy.optimize import minimize, basinhopping, differential_evolution, brute
import scipy.io as sio
import copy

class AMOGAPE():

    def __init__(self, model, D, P, inputlimits, means, stdevs, startpoints = 50, startdata = None):
        """
        model (funcion) is an function taking N x D np.arrays and returning N x P np.arrays
        D (int) is the input dimension
        P (int) is the output dimension
        inputlimits (np.array) is a D x 2 array with the appropriate min and max in each column
        startpoints (int) is the number of LHC generated startpoints. Is ignored if startdata is provided
        startdata (dict) is a dictionary with keys:
            X (np.array) N x D
            Y (np.array) N x P
        """
        self.D = D
        self.P = P
        self.model = model
        self.inputlimits = inputlimits
        self.limdiffs = np.diff(self.inputlimits)
        self.data = {}
        self.tracker = {}

        # These two are only used in in normalpdf() function.
        self.means = means
        self.stdevs = stdevs
        
        self.GAUSSIAN = multivariate_normal(means, np.diag(np.square(stdevs))) #this is not used if 'startdata' is provided.
        
        # These two are used in normalpdf() function only.
        self.covdet = np.linalg.det( np.diag(np.square(stdevs)) )
        self.covinv = np.linalg.inv( np.diag(np.square(stdevs)) )
        
        # Generate data
        if startdata == None:
            # Generate LHC startdata in intervals [0,1] and scale according to given limits
            self.data['X'] = np.zeros([startpoints, self.D])
            n = 0
            
            while startpoints > n:
                lims = self.inputlimits
                point = self.GAUSSIAN.rvs()
                if (lims[0,0] < point[0] < lims[0,1]) & (lims[1,0] < point[1] < lims[1,1]):
                    self.data['X'][n,:] = point
                    n += 1

            try:
                # Check if function accepts np.arrays 
                self.data['Y'] = self.model(self.data['X'])
            except ValueError:
                # Otherwise run loop
                print('Takes only one input at a time, running for starpoint for loop')
                Y = np.zeros([startpoints, P])
                for i in range(startpoints):
                    Y[i,:] = self.model(self.data['X'][i,:].reshape([1,D])).reshape([1,P])
                self.data['Y'] = Y
        else:
            self.data = copy.deepcopy(startdata)
         
        self.N = self.data['Y'].shape[0]

        # Train initial emulators
        self.train_emulators()
        
    
    def train_emulators(self):
        """ Trains an emulator for each output of the physical model
        """
        self.emudict = {'emu'+str(p) : mintGP() for p in range(self.P)}
        for p in range(self.P):
            y = self.data['Y'][:,p].reshape(self.N,1)
            self.emudict['emu'+str(p)].fit(self.data['X'], y)
        
        
    def force_likelihoodnoise(self, force_sigma):
        """ Forces to the likelihood variance to be a certain value for each emulator
        """
        for p in range(self.P):
            self.emudict['emu'+str(p)].likelihood_variance = force_sigma
            
    def force_jitter(self, force_jitter):
        """ Forces to the likelihood variance to be a certain value for each emulator
        """
        for p in range(self.P):
            self.emudict['emu'+str(p)].jitter = force_jitter
            
            
    def euclnorm(self,x):
        return(  np.sqrt( np.sum( np.square(x) ) )  )
    
    def normalpdf(self,x):
        diff = x.reshape(self.D,1) - self.means.reshape(self.D,1) # x-mu
        return ( 1/np.sqrt(((2*np.pi)**self.D) * self.covdet) ) * np.exp(-0.5*np.dot(diff.T, np.dot(self.covinv, diff)))
    
    def A_add_D(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 0.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY += emu_p.predvar(x) 
        self.GEOMETRY = np.nan
        return self.DIVERSITY * self.normalpdf(x)
    
    def A_prod_D(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 1.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY *= emu_p.predvar(x) 
        self.GEOMETRY = np.nan
        return self.DIVERSITY * self.normalpdf(x)
    
    def A_add_G(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 0.
        self.GEOMETRY = 0.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.GEOMETRY += self.euclnorm( emu_p.geometric_info(x) ) 
        self.DIVERSITY = np.nan
        return self.GEOMETRY * self.normalpdf(x)
    
    def A_add_D_add_G(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 0.#1.
        self.GEOMETRY = 0.#1.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY += emu_p.predvar(x) 
            self.GEOMETRY += self.euclnorm( emu_p.geometric_info(x) )  
        self.Aval = self.GEOMETRY*self.DIVERSITY
        return self.GEOMETRY * self.DIVERSITY * self.normalpdf(x)
    
    def A_prod_D_prod_G(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 1.
        self.GEOMETRY = 1.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY *= emu_p.predvar(x) 
            self.GEOMETRY *= self.euclnorm( emu_p.geometric_info(x) )  
        self.Aval = self.GEOMETRY*self.DIVERSITY
        return self.GEOMETRY * self.DIVERSITY * self.normalpdf(x)
    
    def A_add_D_prod_G(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 0.#1.
        self.GEOMETRY = 1.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY += emu_p.predvar(x) 
            self.GEOMETRY *= self.euclnorm( emu_p.geometric_info(x) )  
        self.Aval = self.GEOMETRY*self.DIVERSITY
        return self.GEOMETRY * self.DIVERSITY * self.normalpdf(x)
    
    def A_prod_D_add_G(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 1.
        self.GEOMETRY = 0.#1.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY *= emu_p.predvar(x) 
            self.GEOMETRY += self.euclnorm( emu_p.geometric_info(x) )  
        self.Aval = self.GEOMETRY*self.DIVERSITY
        return self.GEOMETRY * self.DIVERSITY * self.normalpdf(x)
    
    
    def logA(self,x):
        x = x.reshape(1,self.D)
        self.DIVERSITY = 0.
        self.GEOMETRY = 0.
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            self.DIVERSITY += np.log( emu_p.predvar(x) )
            self.GEOMETRY += np.log( self.euclnorm( emu_p.geometric_info(x) )  )
        return np.min([0,self.GEOMETRY]) + self.DIVERSITY
    
    
    def update(self,limitscale,acq):
        """ AMOGAPE update step:
        Minimize acquisition function acq to find x*, compute corresponding model output f(x*),
        add data pair to database
        """

        #np.random.seed(0)
        
        # Define limits of update
        #limits = limitscale*self.inputlimits
        #print('LIM:', np.diff(self.inputlimits)*(1-limitscale)*0.7)
        limits = self.inputlimits*limitscale + np.diff(self.inputlimits)*(1-limitscale)*0.7
        x0 = np.random.sample(self.D)*np.diff(limits).T + limits[:,0]
        bnds = tuple( ((minlim ,maxlim) for minlim, maxlim in limits) )
        
        # Define acquisition function
        negA = lambda x: - acq(x)
        negGradient = autograd.grad( lambda x: - acq(x) )

        print('negA',negA)

        lims = self.inputlimits
        
        self.d = np.random.uniform(low=0.05,high=0.15)
        brute_bnds = tuple( ((minlim + (maxlim-minlim)*self.d ,maxlim - (maxlim-minlim)*self.d) for minlim, maxlim in lims) ); self.brute_bnds = brute_bnds
        self.brutesol = brute( negA, ranges=brute_bnds, Ns=10, finish=False, full_output=True )
        
        x0 = self.brutesol[0]
        # Minimize acquisition function
        sol = minimize( negA, x0, args=(),
                        method='L-BFGS-B',
                        bounds=bnds, jac=negGradient)

        print('sol x:', sol.x)
        
        # Update database
        self.N += 1
        self.data['X'] = np.concatenate( [ self.data['X'], sol.x.reshape([1,self.D]) ], 0 )
        self.data['Y'] = np.concatenate( [ self.data['Y'], self.model(sol.x.reshape([1,self.D])).reshape([1,self.P]) ], 0 )
        
        
        # Train new emulators
        self.old_emudict = copy.deepcopy(self.emudict)
        try:
            self.train_emulators()
        except Exception as e:
            print('train_emulators error with A: ',str(acq),' deleting last update and run again')
            print(e)
            self.data['X'] = self.data['X'][0:-1,:]
            self.data['Y'] = self.data['Y'][0:-1,:]
            self.N -= 1
            self.emudict = self.old_emudict
            self.update(limitscale,acq)

        if type(self.DIVERSITY) is autograd.numpy.numpy_boxes.ArrayBox:
            print('DIV:', self.DIVERSITY._value[0][0])
            self.DIVERSITY = self.DIVERSITY._value[0][0]
        if type(self.GEOMETRY) is autograd.numpy.numpy_boxes.ArrayBox:
            print('GEO:', self.GEOMETRY._value)
            self.GEOMETRY = self.GEOMETRY._value
        self.tracker[self.N] = {'G': self.GEOMETRY, 'D': self.DIVERSITY,
                'lengthscale': [self.emudict['emu'+str(p)].lengthscale for p in range(self.P)] ,
                'likelihood_sigma': [self.emudict['emu'+str(p)].likelihood_variance for p in range(self.P)]}

        ''' Not sure if this line is needed???'''
        #self.tracker[self.N]['x'] = sol.x
        
        print(self.tracker)
        
        return sol.x
    
    def update_LHS(self):
        """ LHS update step:
        Refresh entire database generating N_t new datapoints
        """
        self.N += 1
        self.data['X'] = lhs(D, self.N)*self.limdiffs.T + self.inputlimits[:,0].T
        
        Y = np.zeros([self.N, P])
        for i in range(self.N):
            Y[i,:] = self.model(self.data['X'][i,:].reshape([1,D])).reshape([1,P])
        self.data['Y'] = Y
        # Retrain emulators
        self.train_emulators()
        self.tracker[self.N] = { 'lengthscale': [self.emudict['emu'+str(p)].lengthscale for p in range(self.P)] ,
                'likelihood_sigma': [self.emudict['emu'+str(p)].likelihood_variance for p in range(self.P)]}
        
        
    def update_gauss(self):
        """ LHS update step:
        Refresh entire database generating N_t new datapoints
        """
        self.N += 1
        Xnew = self.GAUSSIAN.rvs()
        lims = self.inputlimits
        while not (lims[0,0] < Xnew[0] < lims[0,1]) & (lims[1,0] < Xnew[1] < lims[1,1]):
            Xnew = self.GAUSSIAN.rvs()
        self.data['X'] = np.concatenate( [ self.data['X'], Xnew.reshape([1,self.D]) ], 0 )
        self.data['Y'] = np.concatenate( [ self.data['Y'], self.model(Xnew.reshape([1,self.D])).reshape([1,P]) ], 0 )
        # Retrain emulators
        self.train_emulators()
        self.tracker[self.N] = { 'lengthscale': [self.emudict['emu'+str(p)].lengthscale for p in range(self.P)] ,
                'likelihood_sigma': [self.emudict['emu'+str(p)].likelihood_variance for p in range(self.P)]}
    
    def test(self,Xt,Yt,pca=0):
        """ Compute RMSE of current emulator on input (Xt,Yt) input grid 
        """
        m = np.shape(Yt)[0]
        Yp = np.zeros((m, self.P))
        
        for p in range(self.P):
            emu_p = self.emudict['emu'+str(p)]
            Yp[:,p] = emu_p.predict(Xt).ravel()
            
        if pca:
                Yp = pca.inverse_transform( Yp )
                
        return np.sqrt( np.mean( np.square( Yp - Yt ) ) )

