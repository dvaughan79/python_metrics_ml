# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:10:47 2014

NOTES: 
1. This class estimates a Tobit censored model.
2. It is important to remember that if I use it for estimating expected durations
   I must think of it as a lognormal model, i.e. y = ln(T).  This must be done 
   prior to calling the functions.
3. In the case of Exp.Duration of clients with the bank:
    a. Censored: those clients who remain with the bank=> t*>\bar{t}, d=1
    b. Uncensored: those clients who left already. d=0
@author: A3940004
"""

import numpy as np
import pandas as pd
from scipy import stats


class tobit_dan: 
    # these are the attributes of the class
    def __init__(self,ymat,xmat,D):
        # before defining attributes clean data
        xmat    = np.asarray(xmat)
        nvar    = xmat.shape[1]
        nobs    = xmat.shape[0]
        ymat    = np.asarray(ymat)
        D       = np.asarray(D)
        # clean and reassign
        data    = pd.DataFrame(np.concatenate((ymat,D,xmat), axis=1)).dropna().values
        xmat    = data[:,2:]
        nobs,nvar = xmat.shape
        ymat    = data[:,0].reshape((nobs,1))
        D       = data[:,1].reshape((nobs,1))
        dof     = nobs-nvar
        # assign
        self.xmat = xmat
        self.D    = D
        self.ymat = ymat
        self.nobs  = nobs
        self.nvar  = nvar
        self.dof   = dof

        # I will use Gauss-Newton.  I need hessian and gradient
        def gradient_tobit_greene(betafull, y,X,D):
            '''
            gradiente del Tobit (right-censored)
            Source: Green, lectures notes in ppt: 16-TobitModels.pptx
            It uses Olsen's reparameterization.
            I'm using broadcasting to optimize
            '''
            # start by assigning the right parameters
            nobs, nvar = X.shape
            theta = betafull[:-1].reshape((nvar,1))
            psi   = betafull[-1][0]
            # compute relevant variables
            a0 = np.dot(X,theta)
            a1 = psi*y + a0 
            lambdai = np.divide(stats.norm.pdf(a1),stats.norm.pdf(a1))
            # ready to compute
            # With respect to theta: notice I'm using broadcasting
            pre11 = np.multiply((1-D),lambdai) - np.multiply(D,a1)
            A11   = (pre11.reshape((nobs,1))*X).sum(axis=0).reshape((nvar,1))
            # With respect to psi
            A21   = np.multiply(D,((1.0/psi - np.multiply(a1,y)))).sum(axis=0).reshape((1,1))
            
            full_grad = np.concatenate((A11, A21), axis=0)
            return full_grad

        # Now the gradient:
        def hessian_tobit_greene(betafull, y,X,D):
            '''
            Hessiano del Tobit (right-censored)
            Source: Green, lectures notes in ppt: 16-TobitModels.pptx
            It uses Olsen's reparameterization.
            '''
            # start by assigning the right parameters
            nobs, nvar = X.shape
            theta = betafull[:-1].reshape((nvar,1))
            psi   = betafull[-1][0]
            # compute relevant variables
            a0 = np.dot(X,theta)
            a1 = psi*y + a0 
            lambdai = np.divide(stats.norm.pdf(a1),stats.norm.pdf(a1))
            deltai  = np.multiply(-lambdai, a1+lambdai)
            # Ready to compute each partition
            # A11
            pre11 = np.multiply((1-D),deltai) - D
            A11   = np.dot(X.T,pre11.reshape((nobs,1))*X)
            # A12
            A12   = (np.multiply(-D, y).reshape((nobs,1))*X).sum(axis=0).reshape((nvar,1))
            # A22 
            A22   = np.multiply(-D,(1.0/psi**2) + np.power(y,2)).sum(axis=0).reshape((1,1))
            # ready to concatenate
            A1     = np.concatenate((A11,A12),axis=1)
            A2     = np.concatenate((A12.T,A22),axis=1)
            Amat   = np.concatenate((A1,A2),axis=0)
            return Amat
        #-----------------------------------
        # Ready to run Gauss-Newton
        #-----------------------------------
        MaxIters = 1000
        dist = 1
        beta_init = np.ones((nvar+1,1))
        counter = 0
        while dist > 0.0001 and counter < MaxIters:
            # get the inverse of the Hessian
            #H      = hessian_tobit_greene(beta_init, ymat,xmat,D)
            #L      = np.linalg.cholesky(H)
            #Linv   = np.linalg.inv(L)
            #hess_inv = np.dot(Linv.T, Linv)
            hess_inv = np.linalg.inv(hessian_tobit_greene(beta_init, ymat,xmat,D))
            # Get the gradient
            grad_inv = gradient_tobit_greene(beta_init, ymat,xmat,D)
            # Newton iteration
            beta_new = beta_init - np.dot(hess_inv, grad_inv)
            # compute distance
            dist = np.max(np.abs(beta_new-beta_init))
            # update estimate and counter
            beta_init = beta_new.copy()
            counter +=1
        # Get my original parameters back:
        # Done with this:
        sigmahat = 1/beta_new[-1]
        betahat  = -beta_new[:-1]*sigmahat
        fullbeta = np.concatenate((betahat,sigmahat.reshape((1,1))),axis=0)
        # Before transforming back to my original parameters I need to compute
        # the likelihood
        def likeli_fn(beta_full,y,X,D):
            '''
            Tobit Likelihood: I have already transformed the data
            '''             
            nobs, nvar = X.shape
            beta  = beta_full[:-1].reshape((nvar,1))
            sigma = beta_full[-1][0]
            # ready to compute everything
            z = (y-np.dot(X,beta))/sigma
            c1 = np.multiply((1-D),np.log(stats.norm.cdf(z)))
            c2 = np.multiply(D,np.log((1.0/sigma)*stats.norm.pdf(z)))
            likeli = c1 + c2
            return likeli.sum()
            
        self.betahat = betahat
        self.sigma   = sigmahat        
        self.likeli_hat = likeli_fn(fullbeta,ymat,xmat,D)
    

    #----------------------------------------
    # these are the methods of the class 
    #----------------------------------------
    def yest(self): 
         '''Returns yhat = E(y|x,d).  Expected value of right-censored normal r.v.
            Following derivation follows Greene,Th.22.3, p.763
            --------------------------------------------------------
            E(t*) = Pr[d=0]E(t*|t*=t) + Pr[d=1]E(t*|t*>t)
                  = \Phi(t)t + (1-\Phi(t-xb))E(t*|t*>t)
                  = \Phi(t)t + (1-\Phi(t-xb))[\mu + \sigma \lambda(xb)]
                  \lambda(xb) = f(x\beta)/(1-F(x\beta))
            --------------------------------------------------------
         NOTE: this is relevant when y is not in logs.'''
         sigma   = self.sigma[0]
         xbeta   = np.dot(self.xmat,self.betahat)
         ymat    = self.ymat
         alpha   = (self.ymat-xbeta)/self.sigma
         lamb_fn = stats.norm.pdf(alpha)/(1-stats.norm.cdf(alpha))
         term1   = stats.norm.cdf(alpha)*ymat
         term2   = (1-stats.norm.cdf(alpha))*(xbeta + sigma*lamb_fn)
         yest_fs = np.asarray(term1) + np.asarray(term2)
         # if individual is dead (D=1) use observed.  Else use estimated
         return self.ymat*self.D + yest_fs*(1-self.D)

    def yest_var(self): 
         '''Returns Var(yhat) 
            Following derivation follows Greene,Th.22.3, p.763
            I need the variance because I need to estimate expected duration
            for a right-censored log-normal random variable
            --------------------------------------------------------
            Var(t*) = \sigma^2(1-\Phi)[(1-\delta) + (\alpha-\lambda)^2\Phi]
            with \delta = \lambda^2 - \lambda \alpha
            --------------------------------------------------------
         NOTE: this is relevant when y is not in logs.'''
         sigma   = self.sigma
         xbeta   = np.dot(self.xmat,self.betahat)/sigma
         alpha   = (self.ymat-xbeta)/self.sigma
         lamb_fn = stats.norm.pdf(alpha)/(1-stats.norm.cdf(alpha))
         delta   = lamb_fn*(lamb_fn-alpha)
         term1   = (sigma**2)*(1-stats.norm.cdf(alpha))
         term2   = (1-delta) + (alpha - lamb_fn)**2 * stats.norm.cdf(1-alpha) 
         return term1 * term2

    def yest_lognormal(self): 
         '''Returns yhat = E(ln y|x,d).  Expected value of truncated lognormal
         NOTES: 1. If y is time, we need to use a lognormal model, i.e. y = ln(T)
                2. If E(y) = \exp{\mu + \sigma^2/2}, where $\mu = E(ln y)$.
                3. As in yest() I have to take care of the right-censoring.
         '''
         mu     = self.yest()
         sigma2 = self.yest_var()
         return np.exp(mu + 0.5*sigma2)

    #    def R2(self):
    #        ''' Compute Ben-Akiva Lermon (Green, p.684)'''
    #        prer2  = (np.asarray(self.ymat)*np.asarray(self.pest()) 
    #            + np.asarray((1-self.ymat))*np.asarray((1-self.pest())))
    #        R2bl   = (1/self.nobs)*prer2.sum()
    #        return R2bl

    def VCV_sph(self):
        ''' compute covariance matrix now (following Wooldridge, p.526)'''
        Amat = np.zeros((self.nvar+1,self.nvar+1))
        for i in xrange(self.nobs):
            # general parameters:
            gamma = self.betahat/self.sigma
            # parameters needed:
            # machine epsilon
            eps = np.finfo(float).eps
            x_i = np.reshape(self.xmat[i,:],(1,self.nvar))
            xbeta = np.dot(x_i,gamma)
            phi = stats.norm.pdf(xbeta) + eps
            Phi = stats.norm.cdf(xbeta) - eps
            a = -(self.sigma)**(-2)*(xbeta*phi - (phi**2/(1-Phi))-Phi)
            b = (self.sigma)**(-3)*(xbeta**2*phi + phi - (xbeta*phi**2)/(1-Phi))/2
            c = -(self.sigma)**(-4)*(xbeta**3 * phi + xbeta*phi - (xbeta*phi**2)/(1-Phi)-2*Phi)/2
            # Ready to compute Amat
            a11 = a*np.dot(x_i.T,x_i)
            a12 = b*x_i.T
            a21 = b*x_i
            a22 = c
            Amat = Amat + np.concatenate(
                (np.concatenate((a11,a21),axis=0),np.concatenate((a12,a22),axis=0))
                ,axis=1)

        return np.linalg.inv(Amat)

    def tstat_sph(self):
        ''' Compute corresponding tstatistics'''
        vcv  = self.VCV_sph()
        se   = np.reshape(np.asarray(np.sqrt(np.diag(vcv[:self.nvar,:self.nvar]))),(self.nvar,1))
        beta = np.asarray(self.betahat)
        return beta/se
        
    def pval_sph(self):
        ''' Compute corresponding pvalues'''
        dof       = self.dof
        pval_sph = 2*(1-stats.t.cdf(abs(self.tstat_sph()),dof));
        return pval_sph

