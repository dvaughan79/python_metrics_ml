# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 09:05:53 2015

@author: a3940004
"""

# -*- coding: utf-8 -*-
"""
--------------------
Created on Thu Jul 24 13:37:39 2014
This function checks for perfect multicollineratity
--------------------
Notes:
Input:  Xmat
Output: Xout, ind_x
Xout: matrix with linearly independent columns of Xmat
ind_x: indicator (1/0) for included variables
@author: A3940004
--------------------
"""
import numpy as np

def multicol_new(Xmat):
    Xmat = np.asarray(Xmat)
    nx,mx = Xmat.shape
    # get a copy of Xmat
    xtest = np.asarray(Xmat).copy()
    # initialize xout with only one column
    xout  = np.zeros((nx,1))
    ind_x = np.zeros((mx,1))
    # ready to start looping over columns
    # chck that it is indeed the case that not full rank
    rank_x = np.linalg.matrix_rank(Xmat)
    if rank_x==mx:
        # if full-rank: just output same matrix
        xout  = Xmat
        ind_x = np.ones((mx,1))
    # if not full-rank, continue below    
    else: 
        for v in range(mx):
            if np.linalg.matrix_rank(Xmat[:,:v+1]) < v+1:
                print 'bad column is: ', v
    #return xout,ind_x
