# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 18:43:46 2014
@author: A3940004
--------------------------------------
DATE: 25/09/2014
Here I want to compute and graph a dendogram.
For ease of computation, I start with an initial grouping that I get from, eg.
cluster analysis.
I then start grouping each cluster until I have one whole cluster.
"""
import numpy as np
import pandas as pd
import missing as mi
import olsdan_new as ols
from scipy import stats

class dendogram: 
    # these are the attributes of the class
    def __init__(self,Xmat,group,d):
        '''
        Parameters:
        1. Xmat: matrix of observations
        2. group: initial partition
        3. d: 1: Euclidean, 2: Max
        --------------------
        Output:
        DendGroup: Nobs x (groupings for each iteration)
        DendDista: Groups x Distance
        '''
        # before defining attributes clean data
        nobs,nvars = Xmat.shape
        ind_s   = np.arange(nobs).reshape((nobs,1))
        data    = mi.missing(np.concatenate((Xmat,group,ind_s), axis=1))
        xmat    = np.asarray(data[:,:-2])
        N       = xmat.shape[0]
        group1  = np.asarray(data[:,-2])
        ind_s   = np.asarray(data[:,-1],dtype=int).flatten()
        # So I now know if some observation had missing data
        self.Xmat_ori = Xmat
        self.xmat  = xmat
        self.group_ori = group1
        self.nobs  = nobs
        self.N     = N
        self.nvars = nvars
        self.ind  = ind_s
        self.distance = d
        #------------------------------------------------------
        # RUN ALGORITHM HERE
        #------------------------------------------------------
        # First: check that my groups go from 0,1,...
        # 1. Unique values and sort them
        newgroups  = pd.Series(group1.flatten()).unique()
        newgroups.sort()
        # 2. Eliminate NaNs from this Index
        newgroups  = newgroups[np.isnan(newgroups)==False]
        # 3. I want to replace newgroups with DF index
        pregroup  = pd.DataFrame(newgroups)
        to_repl   = pregroup.index
        val_rep   = pregroup.values
        pregroup1 = pd.DataFrame(group1)
        group1   = pregroup1.replace(val_rep, to_repl)
        
        Xmat_pos = pd.DataFrame(np.concatenate((group1,xmat),axis=1))
        GrpMat   = group1
        # iterate until I have only one group:
        iter = 0
        max_grp = newgroups.shape[0]
        mas_grp0 = max_grp
        NetMat = np.zeros((max_grp,max_grp))
        while max_grp>1:
            #-------------------------------------------
            print('iter=',iter, 'max_grp=',int(max_grp))
            #-------------------------------------------
            #  since groups go from 0 to max_grp:
            WBVar        = np.zeros((max_grp,max_grp))
            RankMat      = np.zeros((max_grp,3))
            RankMat[:,:] = np.nan
            # Compute distance between groups
            for g in newgroups:
                # use only those in group g:
                X_g = np.asarray(Xmat_pos[Xmat_pos[0]==g])
                X_g = X_g[:,1:]
                # second loop: over clusters
                for l in newgroups:
                    X_l = np.asarray(Xmat_pos[Xmat_pos[0]==l])
                    X_l = X_l[:,1:]
                    # I need distances for each var across i,i' \in g,l
                    # DMat has number of Ng,Nl as dimensions, feach nvar.
                    DMat = np.zeros((X_g.shape[0],X_l.shape[0],nvars))
                    for v in range(nvars):
                        # compute D(i,i'), for (i,i') \in g
                        if d==1:
                            DMat[:,:,v] = (X_g[:,v].reshape((X_g.shape[0],1))
                                     - X_l[:,v].reshape((1,X_l.shape[0])))**2
                        elif d==2:
                            DMat[:,:,v] = np.abs(X_g[:,v].reshape((X_g.shape[0],1))
                                     - X_l[:,v].reshape((1,X_l.shape[0])))
                    # ready to compute inter-cluster variance:
                    if g==l: #within-variance, to avoid double counting
                        if d==1:
                            WBVar[g,l] = 0.5*np.sqrt(DMat.sum(axis=2).sum())
                        elif d==2:
                            WBVar[g,l] = 0.5*(DMat.max(axis=2).sum())
                    else:  # D(A,B) = min_[i \in A, j \in B] d_{ij}
                        if d==1:
                            WBVar[g,l] = np.sqrt(DMat.sum(axis=2)).min()
                        elif d==2:
                            WBVar[g,l] = (DMat.max(axis=2)).min()
            # I need to make diagonals very large
            WBVar1 = ((1-np.eye(max_grp))*WBVar + 
                        np.eye(max_grp)*2*WBVar.max())
            # compute distance between groups and sort them
            # first column: original groups
            RankMat[:,0] = np.arange(max_grp)
            # second column: groups that minimize distance
            RankMat[:,1] = np.argmin(WBVar1,axis=1)
            # thir column: minimum distance
            RankMat[:,2] = np.min(WBVar1,axis=1)
            # sort on distance
            mind  = np.argsort(RankMat[:,2])
            RankMat = RankMat[mind,:]
            # I can get group classes now:
            for g in RankMat[:,0]:
                h    = RankMat[g,1]
                dmat = RankMat[g,2]
                NetMat[RankMat[g,0],h] = dmat
                
            NetMat = NetMat + NetMat.T
            NN = (np.linalg.matrix_power((NetMat>0)*1,2)>0)*1
            # I now have to loop to find the right pair for each cluster
            # So if I have a pair for which both are minimizers of each other
            # then put them together.  Otherwise, live them alone and continue 
            # iteration
            for g in range(max_grp):
                # if min d_{g}<= min d{argmin{g}}: join g and argmin g
                if RankMat[g,1]<=RankMat[int(RankMat[g,0]),1]:
                    RankMat[g,2] = g
                    RankMat[RankMat[g,0],2] = g
                else:
                    RankMat[g,2] = g
            # get an index for previous groups
            RankMat[:,3]  = np.arange(max_grp)
            # replace old group1 by new group1
            to_repl   = RankMat[:,2]
            val_rep   = RankMat[:,3]
            pregroup1 = pd.DataFrame(group1)
            group1    = pregroup1.replace(val_rep, to_repl)
            group1    = group1.iloc[:,0]
            # Verify that I start with 0
            newgroups  = group1.unique()
            newgroups.sort()
            newgroups  = newgroups[np.isnan(newgroups)==False]
            # 3. I want to replace newgroups with DF index
            pregroup  = pd.DataFrame(newgroups)
            to_repl   = pregroup.index
            val_rep   = pregroup.values
            pregroup1 = pd.DataFrame(group1)
            # update group1
            group1    = pregroup1.replace(val_rep, to_repl)
            max_grp   = newgroups.shape[0]
            Xmat_pos  = pd.DataFrame(np.concatenate((group1,xmat),axis=1))
            newgroups = group1.iloc[:,0].unique()
            newgroups  = newgroups[np.isnan(newgroups)==False]
            newgroups.sort()
            iter += 1
            # save groups
            GrpMat = pd.concat((GrpMat,group1),axis=1)
            
            
        self.GrpMat = GrpMat
