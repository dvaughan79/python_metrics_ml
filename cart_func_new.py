# -*- coding: utf-8 -*-
"""
This is a copy my cart_func:
1. I know that is working
2. I want to try several improvements
    A. Dealing with categorical regressors
    B. Even if I'm done exploring one branch because of sample size, I
"""

import numpy as np
import pandas as pd
import olsdan as ols
import missing as mi

def search_fun(yy,xmat,GridDense):
    '''
    Objective: This function searches for the optimal xk that minimizes SSR for the regression
    of y on x_j, restricted to the region reg. 
    --> It is called by cart_func
    -------------------------------------------
    Arguments:
    yy:          dependent variable 
    xmat:        my matrix of regressors
    GridDense:   How dense will the linspace on x_j be
    -------------------------------------------
    Returns: var_min,thr_min,ssr_min, SSRMat
    '''
    nobs,nvar = xmat.shape
    y1 = yy.reshape((nobs,1))
    # initilize SSRMat
    SSRMat = np.zeros((GridDense, nvar))
    one_vec = np.ones((nobs,1))
    # Initialize extent of search
    minq, maxq = 5,95
    # I'd like to get an idea of max_ssr: regression with a constant
    olspre = ols.ols_dan(y1,one_vec)
    max_ssr = 10*olspre.ssr()
    for k in range(nvar):
        # select current regressor
        xk   = xmat[:,k].reshape((nobs,1))
        # drop nans to compute quantiles for limits of grid
        prexk = pd.Series(xk.flatten()).dropna()
        q5,q95 = np.percentile(prexk,[minq,maxq])
        # compute grid
        thr_grid = np.linspace(q5, q95,GridDense)
        # Ready to loop over grid
        for g in range(GridDense):
            dumg = (xk>=thr_grid[g]).reshape((nobs,1))
            # note that my constant is reg_search: should only have 1s-NaNs
            xols = np.concatenate((one_vec,dumg),axis=1)
            # check if I have enough observations:
            nrows = mi.missing(np.concatenate((y1,xols),axis=1)).shape[0]
            if nrows>10:
                olsk = ols.ols_dan(y1,xols)
                SSRMat[g,k] = olsk.ssr()
            else: # if I don't have observations:
                SSRMat[g,k] = max_ssr
    # Ready to get minimizer and threshold:
    # This gives me the global minimum
    ssr_min = SSRMat.min()
    # Now the variable that minimizes the ssr
    var_min = SSRMat.min(axis=0).argmin()
    # Now the grid_point where it was minimized
    grd_min = SSRMat[:,var_min].argmin()
    # To recover the actual threshold i need the q5-95 original grid on the argmin
    q5,q95 = np.percentile(xmat[:,int(var_min)],[minq,maxq])
    thr_min = np.linspace(q5,q95,GridDense)[grd_min]
    
    # ready to leave:
    return var_min,thr_min,ssr_min, SSRMat


def cart_func(yvec,xxmat,GridDense, MaxDepth):
    '''
    This function estimates a CART.
    Arguments:
    yy:   Nx1 dependent variable vector
    xmat: matrix with regressors
    '''
    yy   = yvec
    xmat = xxmat
    # Initialize some parameters
    # GridDense = 50  # number of grid points to find optimal thresholds
    # MaxDepth  = 2   #  Number of horizontal levels in the tree--> affects total number of nodes
    NumNodes  = np.array([2**p for p in range(MaxDepth+1)]).sum()
    # Get size:
    nobs, nvar = xmat.shape
    #----------------------------------------
    # Run level-0 regression outside of loop
    #----------------------------------------
    # initialize with a NaN.  I will remove it after first iteration
    Regions = np.ones((nobs,1))==1
    var_min,thr_min,ssr_min, SSRMat = search_fun(yy,xmat,GridDense)
    ymean   = pd.Series(yy.flatten()).mean()
    # Nodes will be arranged as: [k*,\phi_{k*}, ssrmin*, ymean,side]
    Nodes = []
    Nodes.append([var_min, thr_min, ssr_min, ymean, 0])
    #----------------------------------------
    # Ready to start loop
    #----------------------------------------
    len_nodes = 1
    #while len(Nodes) < NumNodes:
    while len_nodes < NumNodes:
        # -----------------------------------------
        # First: assign parent and child indicators
        # -----------------------------------------
        current_child = len_nodes
        # Now parent:
        if current_child % 2 != 0:  # odd number
            current_par   = int(0.5*(current_child-1))
        else: #even number
            current_par   = int(0.5*(current_child-2))
        # -----------------------------------------
        # this said, I can get current region using the recursion R_k = R_{par(k)} \cap left_side(par_k) or right_side(par_k)
        # -----------------------------------------
        region_park = Regions[:,current_par]
        if len_nodes==1:
            par_k       = Nodes[current_par][0]
            phi_k       = Nodes[current_par][1]
            nodes_par   = Nodes[current_par]
        else:
            par_k       = Nodes[current_par][-1][0]
            phi_k       = Nodes[current_par][-1][1]
            nodes_par   = Nodes[current_par]
        # ------------
        # LEFT SIDE
        # ------------
        left_bool    = (xmat[:,par_k]<phi_k)
        cur_reg_left = np.multiply(region_park,left_bool)
        y1 = pd.Series(yy.copy().flatten())
        y1.loc[cur_reg_left==False] = np.nan
        y1 = np.asarray(y1).reshape((nobs,1))
        # search on the left
        var_min,thr_min,ssr_min, SSRMat = search_fun(y1,xmat,GridDense)
        ymean   = pd.Series(y1.flatten()).mean()
        # update nodes and Region
        Nodes.append([nodes_par,[var_min, thr_min, ssr_min, ymean, -1]])
        Regions = np.concatenate((Regions,cur_reg_left.reshape((nobs,1))),axis=1)
        # ------------
        # RIGHT SIDE
        right_bool  = (xmat[:,par_k]>=phi_k)
        cur_reg_rgt = np.multiply(region_park,right_bool)
        y1 = pd.Series(yy.copy().flatten())
        y1.loc[cur_reg_rgt==False] = np.nan
        y1 = np.asarray(y1).reshape((nobs,1))
        # search on the left
        var_min,thr_min,ssr_min, SSRMat = search_fun(y1,xmat,GridDense)
        ymean   = pd.Series(y1.flatten()).mean()
        # Nodes will be arranged as: [k*,\phi_{k*}, ssrmin*, ymean,side]
        # update nodes and Region
        Nodes.append([nodes_par,[var_min, thr_min, ssr_min, ymean, 1]])
        Regions = np.concatenate((Regions,cur_reg_rgt.reshape((nobs,1))),axis=1)
        #----------------
        # update len_nodes
        len_nodes = len(Nodes)
    #----------------------------------------
    # End-of-function
    #----------------------------------------
    return Nodes, Regions


