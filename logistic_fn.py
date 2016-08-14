# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:36:12 2014
Create a logicstic function: to be called from logit_dan
Arguments: xmat,beta, must be of the same dimension
@author: A3940004
"""
import numpy as np

def logit_cdf(xmat,beta):
    nrows,ncols = xmat.shape
    Xbeta = np.asmatrix(xmat)*np.asmatrix(beta)
    return np.exp(Xbeta)/(1+np.exp(Xbeta))
