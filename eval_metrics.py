# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:29:44 2016

Functions for model evaluation of classifiers.
1. ROC
Source:
Tom Fawcett (2006), "An Introduction to ROC Curves", Patter Recognition Letters
            27

@author: a3940004
"""
import numpy as np

def roc_curve(f_scores, observed_class):
    '''
    Compute the roc curve using Fawcett (2006).
    INPUTS:
    f_scores: obtained from a model
    observed_class: true class for the sample
    OUTPUT:
    ROC_CURVE: set of x,y points for each probability threshold
    '''
    N = f_scores.shape[0]
    rand_asig = observed_class
    prob_sort = np.sort(f_scores)[::-1]
    ROCmat = np.zeros((N,2))
    # initialize threshold, starting from zero and going to the largest possble
    fprev = 0
    # initialize the  number of true and false positives
    fp= tp = 0
    for i in range(N):
        if prob_sort[i]!= fprev:
            ROCmat[i,:] = [1.0*fp/(rand_asig==0).sum(), 1.0*tp/(rand_asig==1).sum()]
            #update fprev
            fprev = prob_sort[i]
        # now update fp, tp
        if rand_asig[i]==1:
            tp +=1
        else:
            fp +=1
    return ROCmat
    
    
def trapezoid_area(x1,x2,y1,y2):
    '''
    Compute the area of a trapezoid given to coordinate points X,Y
    '''
    base   = np.abs(x1-x2)
    height = (y1+y2)/2.0
    return base*height
    
def auc_metric(f_scores, observed_class):
    '''
    Compute the Area Under the Curve following Fawcett (2006).
    INPUTS:
    f_scores: obtained from a model
    observed_class: true class for the sample
    OUTPUT:
    AUC: area under the curve
    '''
    Nobs = f_scores.shape[0]
    rand_asig = observed_class
    prob_sort = np.sort(f_scores)[::-1]
    # initialize threshold, starting from zero and going to the largest possble
    fprev = 0
    # initialize the  number of true and false positives
    fp= tp = 0
    fp_prev = tp_prev = 0
    AUC = 0
    # tOTAL NUMBER OF TRUE POSITIVES AND NEGATIVES
    N = (rand_asig==0).sum()
    P = (rand_asig==1).sum()
    for i in range(Nobs):
        if prob_sort[i]!= fprev:
            AUC += trapezoid_area(fp, fp_prev, tp, tp_prev)
            fprev = prob_sort[i]
            fp_prev = fp
            tp_prev = tp
        if rand_asig[i]==1:
            tp +=1
        else:
            fp +=1
    # done with that loop
    AUC += trapezoid_area(N,fp_prev, P, tp_prev)
    AUC /= N*P
    return AUC