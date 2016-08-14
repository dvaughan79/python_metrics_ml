# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:36:12 2014
Create a logicstic pdf: to be called from logit_dan
Arguments: xmat,beta, must be of the same dimension
@author: A3940004
"""
import numpy as np

def logit_pdf(xmat,beta):
    Xbeta = np.dot(xmat,beta)
    return np.exp(Xbeta)/((1+np.exp(Xbeta))**2)
