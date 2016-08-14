# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:50:05 2014
This class is used to estimate an exponential, Poisson, count-data model 
using Quasi Maximum Likelihood, as described in Wooldridge, Ch.19.
Y: Nx1 vector of a count-data variable.  
NOTE: this model should be used properly.  A Poisson distribution f(k,\lambda) describes 
the probability of k events ocurring withing the observed \lambda interval.
Examples: number of children in a woman's lifecycle.  Number of credit cards o products
that a client purchases in a time period.
@author: A3940004
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats
from scipy import optimize
import missing as mi


class expon_dan: 
    # these are the attributes of the class
    def __init__(self,Y,Xmat):
        # before defining attributes clean data
        xmat    = np.matrix(Xmat)
        nvar    = xmat.shape[1]
        nobs    = xmat.shape[0]
        ymat    = np.matrix(Ymat)
        # clean and reassign
        data    = mi.missing(np.concatenate((ymat,xmat), axis=1))
        ymat    = data[:,0]
        xmat    = data[:,1:]
        nvar    = xmat.shape[1]
        nobs    = xmat.shape[0]
        dof     = nobs-nvar
        # initialize some vector of parameters:  
        beta0    = np.ones((nvar,1))
        self.xmat = Xmat
        self.ymat = Ymat
        self.nobs  = nobs
        self.nvar  = nvar
        self.dof   = dof

        # Compute betahat here as an attribute shared by the methods
        def likeli_fn(betait):
            '''
            Date: 29/05/2014
            Compute ML function: as described in Wooldridge, Ch.19
            '''
            xmat = np.asarray(self.xmat)
            ymat = np.asarray(self.ymat)
            # BETAIT: includes beta and sigma
            betait = np.reshape(np.asarray(betait),(len(betait),1))
            # machine epsilon
            eps = np.finfo(float).eps
            xbeta = np.dot(xmat,betait)
            
            logli = ymat*xbeta - np.exp(xbeta)
            return -logli.sum()
        # compute betahat from here
        prebeta = optimize.fmin(likeli_fn, x0=beta0,full_output=1)
        prebeta = prebeta[0]
        self.betahat = prebeta

        self.likeli_hat = prebeta[1]
    
#
#    #----------------------------------------
#    # these are the methods of the class 
#    #----------------------------------------
#    def yest(self): 
#         '''Returns yhat = E(y|x,d).  Expected value of truncated normal'''
#         xbeta   = np.dot(self.xmat,self.betahat)/self.sigma
#         lamb_fn = stats.norm.pdf(xbeta)/stats.norm.cdf(xbeta)
#         yest    = stats.norm.cdf(xbeta)*(xbeta + self.sigma*lamb_fn)
#         return yest
#
#    #    def R2(self):
#    #        ''' Compute Ben-Akiva Lermon (Green, p.684)'''
#    #        prer2  = (np.asarray(self.ymat)*np.asarray(self.pest()) 
#    #            + np.asarray((1-self.ymat))*np.asarray((1-self.pest())))
#    #        R2bl   = (1/self.nobs)*prer2.sum()
#    #        return R2bl
#
#    def VCV_sph(self):
#        ''' compute covariance matrix now (following Wooldridge, p.526)'''
#        Amat = np.zeros((self.nvar+1,self.nvar+1))
#        for i in xrange(self.nobs):
#            # general parameters:
#            gamma = self.betahat/self.sigma
#            # parameters needed:
#            # machine epsilon
#            eps = np.finfo(float).eps
#            x_i = np.reshape(self.xmat[i,:],(1,self.nvar))
#            xbeta = np.dot(x_i,gamma)
#            phi = stats.norm.pdf(xbeta) + eps
#            Phi = stats.norm.cdf(xbeta) - eps
#            a = -(self.sigma)**(-2)*(xbeta*phi - (phi**2/(1-Phi))-Phi)
#            b = (self.sigma)**(-3)*(xbeta**2*phi + phi - (xbeta*phi**2)/(1-Phi))/2
#            c = -(self.sigma)**(-4)*(xbeta**3 * phi + xbeta*phi - (xbeta*phi**2)/(1-Phi)-2*Phi)/2
#            # Ready to compute Amat
#            a11 = a*np.dot(x_i.T,x_i)
#            a12 = b*x_i.T
#            a21 = b*x_i
#            a22 = c
#            Amat = Amat + np.concatenate(
#                (np.concatenate((a11,a21),axis=0),np.concatenate((a12,a22),axis=0))
#                ,axis=1)
#
#        return np.linalg.inv(Amat)
#
#    def tstat_sph(self):
#        ''' Compute corresponding tstatistics'''
#        vcv  = self.VCV_sph()
#        se   = np.reshape(np.asarray(np.sqrt(np.diag(vcv[:self.nvar,:self.nvar]))),(self.nvar,1))
#        beta = np.asarray(self.betahat)
#        return beta/se
#        
#    def pval_sph(self):
#        ''' Compute corresponding pvalues'''
#        dof       = self.dof
#        pval_sph = 2*(1-stats.t.cdf(abs(self.tstat_sph()),dof));
#        return pval_sph
#
