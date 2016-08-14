# -*- coding: utf-8 -*-
"""
Created on 13-01-2016

@author: A3940004
"""
import sys
#import os.path
sys.path.append('C:\\Users\\a3940004.Edificios\\Documents\Python Scripts')
import numpy as np
import pandas as pd
#from scipy import linalg
#import olsdan as ols
#import missing as mi
#import multicol as mc

'''
This module is used to estimate Lasso Linear and Logistic models.
It has four functions:


'''

#------------------------------------------------
# LINEAR LASSO
#------------------------------------------------
def linear_lasso(y,xmat,lambda_t, beta_init):
    '''
    Estimate linear lasso.
    I have two options:
    1. Y and Xmat have already been standardized
    2. Y and Xmat were demeaned, but not standardized
    Note: demeaning is necessary because we don't want to penalize the constant
    '''
    nobs, nvar = xmat.shape
    # Initial choice: zeros  -> consistent with a "very" large Lambda
    beta_actual = beta_init.reshape((1,nvar))
    dist = 1
    counter = 0
    maxiter = 10000
    # loop externo: si no hay convergencia continúe
    while dist>0.001 and counter<maxiter: 
        # Loop interno: ciclo con los regresores
        beta_old = beta_actual.copy()
        for k in range(nvar):
            # Residuos parciales excluyendo k
            inc_k    = np.setdiff1d(np.arange(nvar),np.array([k]))
            xmat_k   = xmat[:,inc_k]
            beta_k   = beta_actual[0,inc_k]
            resids_k = y - np.dot(xmat_k, beta_k.reshape((nvar-1,1)))
            #-------------------------------------------
            # Simple OLS: is just $\epsilon_{¬k}'x_k/N$: if Y,X are standardized
            #-------------------------------------------
            #beta_star_k = np.dot(resids_k.T,xmat[:,k].reshape((nobs,1)))/(1.0*nobs*x_2)
            #-------------------------------------------
            # Standard OLS: is just $\epsilon_{¬k}'x_k/x_k'x_k$: if Y,X are not standardized
            #-------------------------------------------
            # NOTE: if x was standardized I get the same results, since \sum_i x_i^2 = N s^2 = N
            #-------------------------------------------
            x_2      = np.dot(xmat[:,k].reshape((1,nobs)), xmat[:,k].reshape((nobs,1)))[0]
            beta_star_k = np.dot(resids_k.T,xmat[:,k].reshape((nobs,1)))/x_2
            # Soft-thresholding
            beta_actual[0,k] = np.sign(beta_star_k)*np.max([0,np.abs(beta_star_k)-lambda_t])
        # Actualicemos distancia:
        dist = np.max(np.abs(beta_actual-beta_old))
        counter +=1
    # fin del loop
    return beta_actual
    
    
# SECOND FUNCTION: LOGIT LASSO    
def logit_lasso(ymat,xmat, lambda_t, beta_init):
    '''
    This function computes the Lasso for the Logit as in Friedman, et.al. (2010), 
                    "Regularizatio Paths"...
    yy: binary dependent variable
    xmat: regressors, excluding constant, so xmat must have been demeaned/standardized
    lambda_t: current penalization constant
    beta_init: user-provided initial vector.  It is useful to compute the whoe
                path of lasso coefficients using warm-starts.
    '''
    nobs, nvar = xmat.shape
    # Initialize parameters
    dist = 1
    critval = 1e-4
    counter = 0
    MaxIters = 100
    # outer loop: I need to get a Newton-Update of the IRLS for the logit
    # I need to initialize beta_old
    beta_new = beta_init.reshape((nvar,1))
    #DistMat = np.zeros((MaxIters,1))
    while dist>critval and counter<MaxIters:
        beta_old = beta_new.copy().reshape((nvar,1))
        # -------------------------------------------
        # Given beta_old, I need to set up the linear model z = xbeta + w^{-1}(y-p), x,w
        # -------------------------------------------
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
        # Ready to transform the model:
        wmat = np.sqrt(PreW)
        z_star = np.multiply(wmat,z)
        x_star = wmat*xmat
        # I need to demean x_star
        #z_star = z_star - z_star.mean()
        #x_star = 
        #--------------------------------------
        # From here it's just the lasso on z_star and x_star
        #--------------------------------------
        beta_new = linear_lasso(z_star,x_star, lambda_t, beta_old)
        # Actualicemos distancia:
        dist = np.max(np.abs(beta_new.reshape((nvar,1))-beta_old))
        #DistMat[counter] = dist
        #--------------------------------------
        # update counter
        #--------------------------------------
        counter +=1
    return beta_new, counter


# I now want to get to full path
def lasso_logit_path(ymat,xmat,lambda_max, lambda_min= 0.0000001, G=10):
    '''
    This function computes the whole path of the Lasso for the Logit.
    ymat: dependent variable (1/0)
    xmat: independent variables (already standardized)
    lambda_min: by default is something very small, close to zero
    lambda_max: user must provide it, large enough for everything to be zero
    G: number of grid points.  Default is 10
    ----------------------------------------------------
    Note: I'm using warm starts
    The only thing that is interesting is that I start with 
    an \beta_{00} = 0, and then \beta_{0,k} = \beta_{opt, k-1}
    '''
    nobs, nvar = xmat.shape
    lambda_pth = np.linspace(lambda_min, lambda_max, G)
    BetaPath = np.zeros((G,nvar))
    for g in range(G):
        if g == 0:
            # I start from very large lambda's
            beta_init = np.zeros((nvar,1))
        else:
            beta_init = beta_new.reshape((nvar,1))
        lambda_it = lambda_pth[-1-g]
        # Call my function 
        beta_new, count  = logit_lasso(ymat,xmat, lambda_it, beta_init)
        # Save results
        BetaPath[g,:] = beta_new.flatten()
        
    return BetaPath
    
# Repeat for the linear case
def lasso_linear_path(ymat,xmat,lambda_max, lambda_min= 0.0000001, G=10):
    '''
    This function computes the whole path of the Lasso for the Logit.
    ymat: dependent variable (1/0)
    xmat: independent variables (already standardized)
    lambda_min: by default is something very small, close to zero
    lambda_max: user must provide it, large enough for everything to be zero
    G: number of grid points.  Default is 10
    ----------------------------------------------------
    Note: I'm using warm starts
    The only thing that is interesting is that I start with 
    an \beta_{00} = 0, and then \beta_{0,k} = \beta_{opt, k-1}
    '''
    nobs, nvar = xmat.shape
    lambda_pth = np.linspace(lambda_min, lambda_max, G)
    BetaPath = np.zeros((G,nvar))
    for g in range(G):
        if g == 0:
            # I start from very large lambda's
            beta_init = np.zeros((nvar,1))
        else:
            beta_init = beta_new.reshape((nvar,1))
        lambda_it = lambda_pth[-1-g]
        # Call my function 
        beta_new  = linear_lasso(ymat,xmat, lambda_it, beta_init)
        # Save results
        BetaPath[g,:] = beta_new.flatten()
        
    return BetaPath
    
    
# Repeat for the linear case
def lasso_linear_path_cv(ymat,xmat,lambda_max, 
                         CV_sam = 5,
                             lambda_min= 0.0000001, G=10,
                             labs = None):
    '''
    This function computes the whole path of the Lasso for the Logit.
    It also performs CROSS-VALIDATION
    ymat: dependent variable (1/0)
    xmat: independent variables (already standardized)
    lambda_min: by default is something very small, close to zero
    lambda_max: user must provide it, large enough for everything to be zero
    G: number of grid points.  Default is 10
    ----------------------------------------------------
    Note: I'm using warm starts
    The only thing that is interesting is that I start with 
    an \beta_{00} = 0, and then \beta_{0,k} = \beta_{opt, k-1}
    '''
    # Get default labs
    nvar = xmat.shape[1]
    if labs == None:
        labs = np.array(['var_'+str(i) for i in range(nvar)])
    # Clean the data first:
    predata = pd.DataFrame(np.concatenate((ymat,xmat),axis=1)).dropna().values
    nobs = predata.shape[0]
    ymat = predata[:,0].reshape((nobs,1))
    xmat = predata[:,1:]
    # ready to start
    nobs, nvar = xmat.shape
    # notes about penalization:
    # when lambda_min = 0 ---> no penalization: OLS
    # lambda_max should be large enough so that coefficients become negligible
    lambda_pth = np.linspace(lambda_min, lambda_max, G)
    BetaPath = np.zeros((G,nvar,CV_sam))
    CVRes    = np.zeros((CV_sam,G))
    # I need cross-validation samples
    ind_grp = np.random.multinomial(1,[1.0/CV_sam]*CV_sam, nobs)
    # outer loop: cross-validation
    for c in range(CV_sam):
        # assign to cross-validations samples
        ind_train = ind_grp[:,c]==0
        n_train   = ind_train.sum()
        ind_valid = ind_grp[:,c]==1
        n_valid   = ind_valid.sum()
        # y and x
        y_train   = ymat[ind_train,:].reshape((n_train,1))
        y_valid   = ymat[ind_valid,:].reshape((n_valid,1))
        x_train   = xmat[ind_train,:]
        x_valid   = xmat[ind_valid,:]
        # For this validation sample I need to save SSR
        beta_new = None
        for g in range(G):
            if g == 0:
                # I start from very large lambda's
                beta_init = np.zeros((nvar,1))
            else:
                beta_init = beta_new.reshape((nvar,1))
            lambda_it = lambda_pth[-1-g]
            # Call my function: with training sample
            beta_new  = linear_lasso(y_train,x_train, lambda_it, beta_init)
            k_beta    = np.max(beta_new.shape)
            # Save beta
            BetaPath[g,:,c] = beta_new.flatten()
            # Compute SSR on validation sample
            yhat_valid = np.dot(x_valid, beta_new.reshape((k_beta,1)))
            ssr_valid  = np.power(y_valid-yhat_valid,2).sum()
            # Save SSR
            CVRes[c,g] = ssr_valid
    
    # save as dataframes
    # Beta was collected for all CV samples, and Grid Points
    # I need to average those on the CV samples
    BetaPath = pd.DataFrame(BetaPath.mean(axis=2),columns=labs)
    # Sae comment for CVRes: CVsamples x Grid.  Average across samples
    CVRes    = pd.DataFrame(CVRes.mean(axis=0),columns=['CV_SSR'])
    return BetaPath, CVRes