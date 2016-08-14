# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:10:47 2014

@author: A3940004
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:15:19 2014
Estimate binomial PROBIT
@author: A3940004
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:56:24 2014
Estimate a logistic regression via ML
@author: A3940004
"""
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats
from scipy import optimize
import missing as mi


class probit_dan: 
    # these are the attributes of the class
    def __init__(self,Ymat,Xmat):
        # before defining attributes clean data
        xmat    = np.matrix(Xmat)
        K       = xmat.shape[1]
        T       = xmat.shape[0]
        ymat    = np.matrix(Ymat)
        data    = mi.missing(np.concatenate((ymat,xmat), axis=1))
        ymat    = data[:,0]
        xmat    = data[:,1:]
        K       = xmat.shape[1]
        T       = xmat.shape[0]
        dof     = T-K
        #        xtx     = xmat.T*xmat
        #        xtxinv  = linalg.inv(xtx)
        #        xty     = xmat.T*ymat
        # initialize some vector of parameters
        beta0    = np.ones((K,1))

        self.xmat = Xmat # Pressure (Pascals) 
        self.ymat = Ymat # Temperature (K)
        # create invXX
        #        self.invXX = xtxinv
        self.dof   = dof
        #        self.xtx   = xtx
        #        self.xty   = xty
        self.nobs  = T
        self.nvar  = K

        # I need to be careful with dummy variables
        xpd = pd.DataFrame(xmat)
        dum_ind = np.zeros((K,3))  # T/F,No.Values,min,max
        for v in range(K):
            dum_ind[v,:] = [len(xpd[v].unique())==2, 
                            np.nanmin(xpd[v].unique()),np.nanmax(xpd[v].unique())]
        self.dum_ind = dum_ind
        
        
        # Compute betahat here as an attribute shared by the methods
        def likeli_fn(betait):
            '''
            Compute ML function
            '''
            ymat = self.ymat
            ymat = np.asarray(ymat)
            xmat = self.xmat
            xmat = np.asarray(xmat)
            betait = np.reshape(np.asarray(betait),(len(betait),1))
            # machine epsilon
            eps = np.finfo(float).eps
            normcdf = stats.norm.cdf(np.dot(xmat,betait))
            logli = (ymat*np.asarray(np.log(normcdf+eps)) + 
                        (1-ymat)*np.asarray(np.log(1-normcdf+eps)))
            return -logli.sum()
        # compute betahat from here
        prebeta = optimize.fmin(likeli_fn, x0=beta0,full_output=1,
                                maxfun=1000000, maxiter=1000000)
        prebeta = prebeta[0]
        prebeta = np.reshape(prebeta,(K,1))        
        self.betahat = prebeta
        
        self.likeli_hat = prebeta[1]
    

    #----------------------------------------
    # these are the methods of the class 
    #----------------------------------------
    def yest(self): 
         '''Returns yhat''' 
         return np.dot(self.xmat,self.betahat)

    def R2(self):
        ''' Compute Ben-Akiva Lermon (Green, p.684)'''
        prer2  = (np.asarray(self.ymat)*np.asarray(self.pest()) 
            + np.asarray((1-self.ymat))*np.asarray((1-self.pest())))
        R2bl   = (1/self.nobs)*prer2.sum()
        return R2bl

    def VCV_sph(self):
        '''Compute vcv without correction for robustnesss:
           vcvmat is just the inverse of the hessian'''
        eps = np.finfo(float).eps
        cdfhat = stats.norm.cdf(self.yest())
        weight_den = cdfhat*(1-cdfhat) + eps
        weight_num = stats.norm.pdf(self.yest())**2
        weights    = weight_num/weight_den
        vcvmat  = np.zeros((self.nvar,self.nvar))
        for i in xrange(self.nobs):
            row    = np.reshape(self.xmat[i,:],(self.nvar,1))
            vcvmat = vcvmat + weights[i]*row*row.T
        
        return np.linalg.inv(vcvmat)

    def pest(self): 
         '''Returns estimated probability''' 
         return stats.norm.cdf(self.yest())
    
    def pest_se(self):
        ''' Compute standard error for estimated probability.  
        Note: I'm doing this for all individuals. Not means.
        '''
        phi2 = stats.norm.pdf(self.yest())**2
        cprod = np.dot(self.xmat,np.dot(self.VCV_sph(),self.xmat.T))
        return phi2*np.reshape(np.diag(cprod),(self.nobs,1))
        

    def tstat_sph(self):
        ''' Compute corresponding tstatistics'''
        se   = np.reshape(np.asarray(np.sqrt(np.diag(self.VCV_sph()))),(self.nvar,1))
        beta = np.asarray(self.betahat)
        return beta/se
        
    def pval_sph(self):
        ''' Compute corresponding pvalues'''
        dof       = self.dof
        pval_sph = 2*(1-stats.t.cdf(abs(self.tstat_sph()),dof));
        return pval_sph

    def marginal(self,xvec=None):
        ''' 
         Marginal Effects (evaluated on the mean): 
         Except for the constant, these are the same as the command in stata:
         (margins, dydx(*) atmeans)
         These are the marginal effect on the probability of D=1.
         NOTE: I have to to care special care of dummy variables
         '''
        # 1. CONTINUOUS: Means,
        if xvec== None:
            xmat   = self.xmat
            meanx  = np.reshape(xmat.mean(axis=0),(self.nvar,1))
        else:
            meanx  = np.reshape(xvec,(self.nvar,1))
        dotx   = np.dot(meanx.T,self.betahat)
        pdfhat = stats.norm.pdf(dotx)
        marg_cont = np.asarray(pdfhat*self.betahat)
        # 2. Discrete: D=1,D=0
        if np.max(self.dum_ind[:,0])==1:
            for v in xrange(self.nvar):
                if self.dum_ind[v,0]==1:
                    meanx1 = meanx
                    meanx1[v] = 1
                    dotx1  = np.dot(meanx1.T,self.betahat)
                    cdf1   = stats.norm.cdf(dotx1)
                    # now 0
                    meanx0 = meanx
                    meanx0[v] = 0
                    dotx0  = np.dot(meanx0.T,self.betahat)
                    cdf0   = stats.norm.cdf(dotx0)
                    # Diff
                    marg_cont[v] = cdf1-cdf0
        return marg_cont


    def marginal_se(self):
        ''' Compute std.error for marginal effects with delta method
            See Green page. 675'''
        # 1. Continuous:    
        xmat   = self.xmat
        meanx  = np.reshape(xmat.mean(axis=0),(self.nvar,1))
        dotx   = np.dot(meanx.T,self.betahat)
        term1  = stats.norm.pdf(dotx)**2
        term2  = np.eye(self.nvar) - (np.dot(self.betahat.T,meanx)*
                                      np.dot(self.betahat,meanx.T))
        marg_vcv  = term1*term2*self.VCV_sph()*term2.T
        marg_cont = np.reshape(np.asarray
                                    (np.sqrt(np.diag(marg_vcv))),(self.nvar,1))
        # 2. Discrete:
        
        if np.max(self.dum_ind[:,0])==1:
            for v in xrange(self.nvar):
                if self.dum_ind[v,0]==1:
                    meanx1 = meanx
                    meanx1[v] = 1
                    dotx1  = np.dot(meanx1.T,self.betahat)
                    pdf1   = stats.norm.pdf(dotx1)
                    pro1   = pdf1*meanx1
                    # now 0
                    meanx0 = meanx
                    meanx0[v] = 0
                    dotx0  = np.dot(meanx0.T,self.betahat)
                    pdf0   = stats.norm.pdf(dotx0)
                    pro0   = pdf0*meanx0
                    # Diff gives \partial F/ \partial \beta
                    Diff = np.reshape(pro1-pro0,(self.nvar,1))
                    # ready to compute matrix product
                    cpro = np.dot(Diff.T,np.dot(self.VCV_sph(),Diff))
                    marg_cont[v] = cpro
        return marg_cont

    def marginal_pval(self):
        ''' Compute corresponding pvalues for marginal effects'''
        tstat     = self.marginal()/self.marginal_se()
        return 2*(1-stats.t.cdf(abs(tstat),self.dof));


    def correct_pred(self):
        ''' Compute the fraction of correctly predicted'''
        c22  = np.mean(np.logical_or(
                np.logical_and(self.pest()<0.5,self.ymat==0),
                np.logical_and(self.pest()>=0.5,self.ymat==1)))
        return c22