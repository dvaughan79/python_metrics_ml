# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:56:24 2014

@author: A3940004
"""
import sys
#import os.path
sys.path.append('C:\\Users\\a3940004.Edificios\\Documents\Python Scripts')
import numpy as np
import pandas as pd
#from scipy import linalg
import olsdan as ols
import missing as mi
#import multicol as mc

'''
This module is used to estimate Generalized Additive Models.
It has three functions:

1. nat_spline_smoother(y,x,K): 
    scatter-plot natural cubic spline smoother.
    Takes two variables, y and x, and K as inputs and smoothes their relation.

2. spline_fitted(x_t,K,y_v,x_v,beta): 
    in case I need to estimate SSR and Yhat
    in a validating sample, this function does the following:
    A. Takes knots and betas from training sample.
    B. Compute independent smoothers for the Xs in the validating sample
        using the knots and betas in A.
    C. I need to estimate the backfitting routine to get yhat.

3. gam_estimator(y_t,xmat_t,K,y_v, xmat_v):
    This function does the full job, but it is fairly simple:
    A. Loop over the xmat_t variables smooth the residuals of Y on the
        remaining ones.
    B. This is done until convergence.
'''

def nat_spline_smoother(y,x,K):
    '''
    ------
    Objective:
    This function estimates a scatterplot natural cubic-spline smoother
    of Y on X with K knots that will end up being quantiles.
    ------
    Args: 
    y: dependent variable: (Nx1)
    x: independent variable : (Nx1) same size as Y, in part. it is a vector
    K: number of knots
    ------
    Returns: fitted values of y_v on validating sample        
    ------
    '''
    # Get number of observations since my output will have the same number:
    N = y.shape[0]
    #----------------------------------------------
    # First, construct knots with percentiles
    knots = np.percentile(x,np.linspace(0,1,K+2)*100)
    knots1 = knots[1:-1]
    # I need to fix the case where I have repeated knots, being careful with first and last
    knots1 = pd.Series(knots1)
    # take care of last:
    ind_dup = pd.DataFrame(knots1).duplicated(take_last=True).values
    ind_dup[0]=False
    knots1.loc[ind_dup] = np.nan
    # take care of first
    ind_dup = pd.DataFrame(knots1).duplicated(take_last=False).values
    ind_dup[ind_dup.shape[0]-1]=False
    knots1.loc[ind_dup] = np.nan
    # With pandas this is very simple:
    #ind_dup = pd.DataFrame(knots1).duplicated(take_last=True).values
    #knots1[ind_dup==True]  = np.nan
    knots1 = np.asarray(knots1.interpolate())
    #----------------------------------------------
    # Construct matrix of regressors, by concatenating over knots
    prex = np.power((x-knots1[0]),3)
    for k in knots1[1:]:
        prex = np.concatenate((prex, np.power((x-k),3)), axis=1)
    prex[prex<0] = 0
    #Xmat = np.concatenate((np.ones((N,1)),x,np.power(x,2),np.power(x,3), prex),axis=1)
    #ols_ncs = ols.ols_dan(y,Xmat)
    #betahat = ols_ncs.betahat().reshape((ols_ncs.nvar,1))
    #y_hat  = np.dot(Xmat,betahat)
    #----------------------------------------------
    # I know that
    # 1. N_{k+2} = d_k(x) - d_{K-1}(x)
    # 2. d_k(x)  = \frac{poscub(x-\pxi_k) - poscub(x-\pxi_K)}{\psi_K-\psi_k}
    #--------------------
    # so I can fix d_{K-1}
    d_K_1 = ((prex[:,-2] - prex[:,-1])/(knots1[-1]-knots1[-2])).reshape((N,1))
    N_mat = np.concatenate((np.ones((N,1)),x),axis=1)
    for k in range(K-2):
        d_k = ((prex[:,k] - prex[:,K-1])/(knots1[K-1]-knots1[k])).reshape((N,1))
        delta_d = d_k - d_K_1
        N_mat = np.concatenate((N_mat,delta_d),axis=1)
    # ready to run OLS
    #    print  "----------------------"
    #    print y.shape, N_mat.shape, np.linalg.matrix_rank(N_mat), Xmat.shape
    #    print  "----------------------"
    ols_ncs = ols.ols_dan(y,N_mat)
    ## Get fitted values:
    betahat = ols_ncs.betahat().reshape((ols_ncs.nvar,1))
    y_hat  = np.dot(N_mat,betahat)
    return x,y_hat, betahat


# Get a function to get the knots for prediction
def spline_fitted(x_t,K,y_v,x_v,beta):
    '''
    Fit spline on a validation sample.  For the training sample it comes directly from
    nat_spline_smoother1
    ------
    Args:
    1. x_t: x in training.  Need it to compute nodes
    2. K  : number of nodes used in training
    3. y_v: y for validating
    4. x_v: x for validating
    5. beta: these are the betas from the scatter-smoother in the training sample
    ------
    Returns: fitted values of y_v on validating sample        
    '''
    # Get number of observations since my output will have the same number:
    N = y_v.shape[0]
    #-------------------------------------------------
    # KNOTS ARE FOUND USING training SAMPLE
    #----------------------------------------------
    # First, construct knots with percentiles
    knots = np.percentile(x_t,np.linspace(0,1,K+2)*100)
    knots1 = knots[1:-1]
    # I need to fix the case where I have repeated knots, being careful with first and last
    knots1 = pd.Series(knots1)
    # take care of last:
    ind_dup = pd.DataFrame(knots1).duplicated(take_last=True).values
    ind_dup[0]=False
    knots1.loc[ind_dup] = np.nan
    # take care of first
    ind_dup = pd.DataFrame(knots1).duplicated(take_last=False).values
    ind_dup[ind_dup.shape[0]-1]=False
    knots1.loc[ind_dup] = np.nan
    # With pandas this is very simple:
    #ind_dup = pd.DataFrame(knots1).duplicated(take_last=True).values
    #knots1[ind_dup==True]  = np.nan
    knots1 = np.asarray(knots1.interpolate())
    #----------------------------------------------
    #-------------------------------------------------
    # The first part of the cubic spline is built with VALIDATING sample
    prex = np.power((x_v-knots1[0]),3).reshape((N,1))
    for k in knots1[1:]:
        prex = np.concatenate((prex, np.power((x_v-k),3).reshape((N,1))), axis=1)
    prex[prex<0] = 0
    #----------------------------------------------
    # I know that
    # 1. N_{k+2} = d_k(x) - d_{K-1}(x)
    # 2. d_k(x)  = \frac{poscub(x-\pxi_k) - poscub(x-\pxi_K)}{\psi_K-\psi_k}
    #--------------------
    # so I can fix d_{K-1}
    d_K_1 = ((prex[:,-2] - prex[:,-1])/(knots1[-1]-knots1[-2])).reshape((N,1))
    N_mat = np.concatenate((np.ones((N,1)),x_v.reshape((N,1))),axis=1)
    for k in range(K-2):
        d_k = ((prex[:,k] - prex[:,K-1])/(knots1[K-1]-knots1[k])).reshape((N,1))
        delta_d = d_k - d_K_1
        N_mat = np.concatenate((N_mat,delta_d),axis=1)
    # Ready to estimate the smoother with training data:
    y_hat  = np.dot(N_mat,beta)
    # Here I only need the estimate
    return y_hat
    
def gam_estimator(*args):
    '''
    Estimate a Generalized Additive Model using:
    1. Backfitting
    2. Natural cubic spline smoother
    y: Dependent variable
    x: Regressors including constant
    K: number of knots
    '''
    # I on ly have two options: (y,x,K) or (y_t,x_t,K,y_v,x_v)
    if len(args) == 3:
        y = args[0]
        x = args[1]
        K = args[2]
        # Clean data from here
        N,nvar = x.shape
        pred = mi.missing(np.concatenate((y.reshape((N,1)),x),axis=1))
        Np   = pred.shape[0]
        y = pred[:,0].reshape((Np,1))
        x = pred[:,1:]
        N,nvar = x.shape
    else:
        # for training set
        y = args[0]
        x = args[1]
        K = args[2]
        # Clean data from here
        N,nvar = x.shape
        pred = mi.missing(np.concatenate((y.reshape((N,1)),x),axis=1))
        Np   = pred.shape[0]
        y = pred[:,0].reshape((Np,1))
        x = pred[:,1:]
        N,nvar = x.shape
        # for validation set
        y_v = args[3]
        x_v = args[4]
        N_v    = x_v.shape[0]
        pred = mi.missing(np.concatenate((y_v.reshape((N_v,1)),x_v),axis=1))
        Np   = pred.shape[0]
        y_v = pred[:,0].reshape((Np,1))
        x_v = pred[:,1:]
        N_v    = x_v.shape[0]
    # initialize dist, the criterion for the while statement
    dist = 1
    crit = 0.001  # stop when differences are smaller than crit
    # Initialize functions---> follow the book, zeros for everything except intercept
    Fmat = np.zeros((N,nvar))
    Fmat[:,0] = np.nanmean(y)*np.ones((N))
    Bmat = np.zeros((K,nvar))
    # ready to start
    while dist>crit:
        # update old_func
        old_func = Fmat.copy()
        # Algorithm is simple: for each regressor i (exc.constant)
        # 1. Create residuals excluding i
        # 2. Apply smoother
        # 3. Demean and compute new differences
        for i in range(1,nvar):
            # for each regressor i I need to estimate a smoother on the residuals:
            # select indices on those excluded:
            ind_i = np.setdiff1d(range(nvar),np.array([i]))
            # first I need the current residuals:
            # Since this is an additive model it is easy:
            res_i = y.reshape((N,1)) - Fmat[:,ind_i].sum(axis=1).reshape((N,1))
            # Now apply smoother of res_i on xmat_i: with K knots
            xl_i,f_i,b_i   = nat_spline_smoother(res_i,x[:,i].reshape((N,1)),K)
            Fmat[:,i]  = f_i.flatten()
            Bmat[:,i]  = b_i.flatten()
        # Done with iteration
        # 1. Demean Fs
        Fmat[:,1:] = Fmat[:,1:] - np.nanmean(Fmat[:,1:],axis=0).reshape((1,nvar-1))
        # Compute square distances across columns and use the max
        dist = ((Fmat[:,1:]-old_func[:,1:])**2).sum(axis=0).max()
    #--------------------------------------------------------
    # DONE: save, yhat and ssr
    yhat = Fmat.sum(axis=1)
    ssr  = np.dot(yhat.T,yhat)
    # This is it if I don't have validating sample
    if len(args) == 3:
        return Fmat, yhat, ssr
    # If I do have passed validating sample info:
    else:
        #--------------------------------------------------------
        # VALIDATING SAMPLE: compute SSR and PREDICT
        #--------------------------------------------------------
        # Steps are similar:
        # 1. Use x_training to compute betas and knots---> spline_fitted(x,K,y_v,x_v,beta)
        # 2. Since scatter-smoothing is bivaraite only (y,x) ---> I still need backfitting
        Fmat_v = np.zeros((N_v,nvar))
        Fmat_v[:,0] = np.nanmean(y_v)*np.ones((N_v))
        # reinitialize dist:
        dist = 1
        # ready to start
        while dist>crit:
            # update old_func
            old_func = Fmat_v.copy()
            for i in range(1,nvar):
                # Exclude i to netout other regressors
                ind_i = np.setdiff1d(range(nvar),np.array([i]))
                # Because of additivity, netting out is simple
                res_i        = y_v.reshape((N_v,1)) - Fmat_v[:,ind_i].sum(axis=1).reshape((N_v,1))
                # Smoother on res_i: ----> spline_fitted(x,K,y_v,x_v,beta)
                beta_i       = Bmat[:,i].reshape((K,1))
                # careful with arguments: x_t,K,y_v,x_v,beta
                f_i          = spline_fitted(x[:,i],K,res_i,x_v[:,i],beta_i)
                Fmat_v[:,i]  = f_i.flatten()
            # Done with iteration
            # 1. Demean Fs
            Fmat_v[:,1:] = Fmat_v[:,1:] - np.nanmean(Fmat_v[:,1:],axis=0).reshape((1,nvar-1))
            # Compute square distances across columns and use the max
            dist = ((Fmat_v[:,1:]-old_func[:,1:])**2).sum(axis=0).max()
        #-----------------------------
        # Done with the loop
        #-----------------------------
        yhat_v = Fmat_v.sum(axis=1).reshape((N_v,1))
        ssr_v  = np.dot(yhat_v.T,yhat_v)[0,0]
        return Fmat, yhat, ssr, yhat_v, ssr_v