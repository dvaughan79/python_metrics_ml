# -*- coding: utf-8 -*-
"""
NOTE 24-09-2015
----------------
1. Here I program the newton-raphson algorithm myself, i.e. iteratively
reweighted least squares
2. In logit_optimize I use the solver
3. I tried both in trial_linprob.ipynb and the work as planned
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
#from scipy import linalg
from scipy import stats
#from scipy import optimize
import missing as mi
import multicol as mc
import olsdan as ols
import logit_cdf as lcdf
import logit_pdf as lpdf
reload(mc)

class logit_dan: 
    # these are the attributes of the class
    #def __init__(self,Ymat,Xmat):
    def __init__(self,*args):
        nargs   = len(args)
        if nargs == 2: # I only included y,X
            Ymat  = args[0]
            Xmat  = args[1]
            names = np.asarray(['X'+str(i) for i in range(Xmat.shape[1])])
        elif nargs == 3:
            Ymat  = args[0]
            Xmat  = args[1]
            names = args[2]
        # before defining attributes clean data
        xmat    = np.asarray(Xmat)
        ymat    = np.asarray(Ymat)
        data    = mi.missing(np.concatenate((ymat,xmat), axis=1))
        prexmat = np.asarray(data[:,1:])
        T       = data.shape[0]
        ymat    = np.asarray(data[:,0]).reshape((T,1))
        # see if I have multicolineality
        xmat, ind_out = mc.multicol(prexmat)
        # get the names and a data frame of xmat with names
        names   = names[ind_out.flatten()==1]
        #xmat_df = pd.DataFrame(xmat, columns=names)
        K       = xmat.shape[1]
        dof     = T-K
        
 

        self.xmat = xmat
        self.ind_out = ind_out
        self.ymat = ymat 
        # create invXX
        #self.invXX = xtxinv
        self.dof   = dof
        #self.xtx   = xtx
        #self.xty   = xty
        self.nobs  = T
        self.nvar  = K
        #self.xmat_df = xmat_df
        self.names = names
        
        # I need to be careful with dummy variables
        xpd = pd.DataFrame(xmat)
        dum_ind = np.zeros((K,3))  # T/F,No.Values,min,max
        for v in range(K):
            dum_ind[v,:] = [len(xpd[v].unique())==2, 
                            np.nanmin(xpd[v].unique()),np.nanmax(xpd[v].unique())]
        self.dum_ind = dum_ind
        
        #----------------------------------------------
        # NEWTON RAPHSON:
        # initialize with ols: I know they're wrong but at least they have 
        # the right signs
        olsini  = ols.ols_dan(ymat,xmat)
        beta_old = np.asarray(olsini.betahat()).reshape((olsini.nvar,1))
        # clean from missing values
        # ready to start algorithm:
        dist = 1
        maxIter = 1
        counter = 0
        while dist>0.0001 and counter<maxIter:
            # start by getting the diagonal
            xbeta = np.dot(xmat,beta_old)
            Pmat  = np.divide(np.exp(xbeta), 1+np.exp(xbeta))
            PreW  = np.multiply(Pmat,1-Pmat)
            # check if zeros
            PreW[PreW<1e-4]   = 0.001
            PreW[PreW>0.9999] = 0.99
            # Y-p
            Y_minusP = ymat - Pmat
            # I don't want to use the diagonal matrix as it becomes prohibitely large with large N
            # I can just use broadcasting:
            w_inv  = 1.0/PreW
            z      = xbeta + np.multiply(w_inv, Y_minusP)
            # Ready to compute my update:  do it by parts
            # I want to use the cholesky decomposition of xtx
            premat = np.dot(xmat.T, PreW*xmat)
            L = np.linalg.cholesky(premat)
            newbeta = np.linalg.solve(L.T, np.linalg.solve(L,np.dot(xmat.T,
                                                                      np.multiply(PreW,z))))       
            dist = np.max(np.abs(beta_old-newbeta))
            # update beta_old
            beta_old = newbeta
            counter +=1

        self.premat = premat
        self.betahat  = newbeta
        # get a dataframe
        betahatdf = pd.DataFrame(newbeta,index=names)
        self.betahatdf = betahatdf
        self.names = names
        self.betainit = olsini.betahat()
        
        self.iterations = counter
        # I need the loglikelihood
        xbeta_opt = np.dot(xmat,newbeta)
        Pvec_opt  = np.exp(xbeta_opt)/(1+np.exp(xbeta_opt))
        p1i = np.multiply(ymat,np.log(Pvec_opt))
        p0i = np.multiply(1-ymat,np.log(1-Pvec_opt))
        logli = np.add(p1i,p0i)
        logli = logli.sum()

        self.likeli_hat = logli
        self.weights = np.multiply(Pvec_opt,1-Pvec_opt)

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
        ''' Computes robust White VCV matrix'''
        xmat = self.xmat
        nobs = xmat.shape[0]
        PreW = (self.weights).reshape((nobs,1))
        premat = np.dot(xmat.T, PreW*xmat)
        return np.linalg.inv(premat)

    def pest(self): 
         '''Returns estimated probability''' 
         return lcdf.logit_cdf(self.xmat,self.betahat)
    
    def pest_se(self):
        ''' Compute standard error for estimated probability.  
        Note: I'm doing this for all individuals. Not means.
        '''
        phi2 = lpdf.logit_pdf(self.xmat,self.betahat)**2
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

    def pval_sph_df(self):
        ''' Compute corresponding pvalues'''
        dof       = self.dof
        names     = self.names
        pval_sph  = 2*(1-stats.t.cdf(abs(self.tstat_sph()),dof))
        pval_sph  = pd.DataFrame(pval_sph, index=names)
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
            meanx  = np.reshape(np.nanmean(xmat,axis=0),(self.nvar,1))
        else:
            meanx  = np.reshape(xvec,(self.nvar,1))
        dotx   = np.dot(meanx.T,self.betahat)
        pdfhat = np.exp(dotx)/(1+np.exp(dotx))**2
        marg_cont = np.asarray(pdfhat*self.betahat)
        #return meanx
        # 2. Discrete: D=1,D=0
        if np.max(self.dum_ind[:,0])==1:
            for v in xrange(self.nvar):
                if self.dum_ind[v,0]==1:
                    meanx1 = meanx
                    meanx1[v] = 1
                    cdf1   = lcdf.logit_cdf(meanx1.T,self.betahat)
                    # now 0
                    meanx0 = meanx
                    meanx0[v] = 0
                    cdf0   = lcdf.logit_cdf(meanx0.T,self.betahat)
                    # Diff
                    marg_cont[v] = cdf1-cdf0
        return marg_cont

    def marginal_df(self):
        return pd.DataFrame(self.marginal(), index = self.names)

    def marginal_se(self,xvec=None):
        ''' Compute std.error for marginal effects with delta method
            See Green page. 675'''
        # 1. Continuous:    
        if xvec== None:
            xmat   = self.xmat
            meanx  = np.reshape(xmat.mean(axis=0),(self.nvar,1))
        else:
            meanx  = np.reshape(xvec,(self.nvar,1))
        dotx   = np.dot(meanx.T,self.betahat)
        logpdf = np.exp(dotx)/(1+np.exp(dotx))**2
        term1  = (logpdf*(1-logpdf))**2
        term2  = (np.eye(self.nvar)+(1-2*logpdf))*(np.dot(self.betahat,meanx.T))
        marg_vcv = term1*term2*self.VCV_sph()*term2.T
        marg_cont = np.reshape(np.asarray
                                    (np.sqrt(np.diag(marg_vcv))),(self.nvar,1))
        # 2. Discrete:
        
        if np.max(self.dum_ind[:,0])==1:
            for v in xrange(self.nvar):
                if self.dum_ind[v,0]==1:
                    meanx1 = meanx
                    meanx1[v] = 1
                    pdf1   = lpdf.logit_pdf(meanx1.T,self.betahat)
                    pro1   = pdf1*meanx1
                    # now 0
                    meanx0 = meanx
                    meanx0[v] = 0
                    pdf0   = lpdf.logit_pdf(meanx0.T,self.betahat)
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

    def marginal_pval_df(self):
        return pd.DataFrame(self.marginal_pval(), index = self.names)


    def correct_pred(self):
        ''' Compute the fraction of correctly predicted'''
        c22  = np.mean(np.logical_or(
                np.logical_and(self.pest()<0.5,self.ymat==0),
                np.logical_and(self.pest()>=0.5,self.ymat==1)))
        return c22