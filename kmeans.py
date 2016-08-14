# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:57:43 2014
Given K groups that the user provides:
1. Using b random initial draws computes Kmeans clusters
2. To choose the best solution (out of the b draws) I minimize the maximum
   within-variance across groups
DATE: 18/09/14
I want to include an inf-norm (max distance).
I need to change the algorithm a bit
@author: A3940004
"""

import numpy as np
import pandas as pd
import missing as mi
import olsdan_new as ols
from scipy import stats

class kmeans: 
    # these are the attributes of the class
    def __init__(self,Xmat,K,B,d):
        '''
        Parameters:
        1. Xmat: matrix of observations
        2. K: Number of clusters the user wants
        3. b: Number of random draws used to initialize algorithm
        4. d: 1: Euclidean, 2: Max
        --------------------
        Output:
        Xmat_ori: with NaNs
        xmat    : without nans
        nobs    : Total with Nans
        N       : observations without nans
        K       : Number of clusters:
        b       : Number of initial random draws
        ind     : Indicator for non-Nan observations
        '''
        # before defining attributes clean data
        nobs,nvars = Xmat.shape
        # This will help me handle missing values
        ind_s   = np.arange(nobs).reshape((nobs,1))
        data    = mi.missing(np.concatenate((Xmat,ind_s), axis=1))
        xmat    = np.asarray(data[:,:-1])
        N       = xmat.shape[0]
        ind_s   = np.asarray(data[:,-1],dtype=int).flatten()
        # So I now know if some observation had missing data
        self.Xmat_ori = Xmat
        self.xmat  = xmat
        self.nobs  = nobs
        self.N     = N
        self.nvars = nvars
        self.K    = K
        self.B    = B
        self.ind  = ind_s
        self.distance = d
        #------------------------------------------------------
        # RUN ALGORITHM HERE
        #------------------------------------------------------
        Xmat = xmat
        KK   = K-1
        # full sample
        #        nobs = self.nobs
        # without missing values:
        N    = int(Xmat.shape[0])
        #        ind_s = self.ind
        
        #---------------------------------------------------
        # NOTE: 04/06/2015
        # This is computationally expensive, since I compute NxN matrices        
        # First: Compute Total Variance, done out-of-loop
        # Since it's independent of number of clusters
        #---------------------------------------------------
        #DMat = np.zeros((N,N,nvars))
        #for v in range(nvars):
        #    if d==1:
        #        DMat[:,:,v] = (xmat[:,v].reshape((N,1))
        #                 - xmat[:,v].reshape((1,N)))**2
        #    elif d==2:
        #        DMat[:,:,v] = np.abs(xmat[:,v].reshape((N,1))
        #                 - xmat[:,v].reshape((1,N)))
        # Now: 1. Sum over vars, and then sum over i,j
        #if d==1:
        #    TotalVar = 0.5*np.sqrt(DMat.sum(axis=2).sum())
        #elif d==2:
        #    TotalVar = 0.5*(DMat.max(axis=2).sum())

        #------------------------------------------------------------------
        # Ready to start loop:
        # First loop: I will have B random initial points
        #             I will select the results with min. within variance
        # Second Loop: Stop until convergence (i.e. no ind switches cluster)
        #------------------------------------------------------------------
        # Initialize some matrices
        IndKmean  = np.zeros((N,B))
        WithVar   = np.zeros((B))
        # Loop over random initial assignments
        for b in range(B):
            # Initialize groups: random assignment
            rnd_ass = pd.Series(np.dot(np.random.multinomial(1,[1.0/K]*K,N),
                                                   np.arange(K)))
            # Loop over fraction of individuals that change
            frc_chg = 1
            ind = 1
            while frc_chg>0.0001:
                print('-------------------------')
                print(u'Iteracion: ',ind)
                print(u'Fraccion de Cambios: ',frc_chg)
                print('-------------------------')
                # Step 1: compute centroid: us Groupby
                #Xmat_pre = pd.DataFrame(np.concatenate((rnd_ass,Xmat),axis=1))
                means   = pd.DataFrame(Xmat).groupby(rnd_ass.values.flatten()).mean()
                medians = pd.DataFrame(Xmat).groupby(rnd_ass.values.flatten()).median()
                #                gm       = means.shape[0]
                #----------------------------------------    
                # Euclidean Distance:
                #----------------------------------------    
                DistMat = np.zeros((N,KK+1))
                if d==1:  # compute Euclidean distances:
                    for g in range(KK+1):
                        # Means are centroids so assign i->G such that min dist
                        # use broadcasting and only one instruction
                        DistMat[:,g] = ((Xmat-means.ix[g].values.reshape((1,nvars)))**2).sum(axis=1)
                elif d==2: #Inf-Norm (max distance)
                    for g in range(KK+1):
                        DistMat[:,g] = np.abs(Xmat-medians.ix[g].values.reshape((1,nvars))).max(axis=1)
                # Get minimizers of distance since these will be
                # my assignments of clusters
                ind_g_new = pd.Series(np.argmin(DistMat,axis=1))
                # Compute fraction of changes:
                frc_chg = np.mean(ind_g_new!=rnd_ass)
                # update rnd_ass
                rnd_ass = ind_g_new
                #-----------------------------------------------------        
                #-----------------------------------------------------        
                # If one group has been eliminated I have two options:
                #-----------------------------------------------------        
                # 1. Include some of the missing group randomly
                # 2. Just keep going and let the minimum variance criterion tell me
                # Stick to option 1
                #---------------
                #rnd_ass = rnd_ass.values.flatten()
                # check if new assignment has correct number of groups
                inc_cnt = rnd_ass.groupby(rnd_ass).count()
                mss_grp = np.setdiff1d(np.arange(K),np.asarray(inc_cnt.index))
                # check if I have any missing groups
                rnd_ass = pd.Series(rnd_ass)
                if mss_grp.shape[0]>0:
                    # I want 5% of random of largest group
                    ind_lg_grp = pd.Series(rnd_ass).loc[rnd_ass==inc_cnt.argmax()].index
                    sze = int(0.05*inc_cnt.max())
                    for gg in mss_grp:
                        ind_choice = np.random.choice(ind_lg_grp,sze)
                        rnd_ass.loc[ind_choice] = gg
                # update ind
                ind +=1

            #----------------------------
            # DONE WITH LOOP
            #----------------------------
            # Save indices
            IndKmean[:,b] = rnd_ass.values.flatten()
            # Compute within variance:
            # \sum_g (n_g-1)s_g ^2/(N-G)
            prem = pd.DataFrame(Xmat).groupby(rnd_ass).count()-1
            posm = pd.DataFrame(Xmat).groupby(rnd_ass).std()
            wvar = (prem*posm).sum().sum()/(N-K)
            WithVar[b] = wvar
        #------------------- END OF WHILE LOOP -----------------------#                
        # so ready to finish: minimize the maximum within group variance:
        min_maxvar = np.argmin(WithVar)
        ind_optimal = IndKmean[:,min_maxvar]
        # now use missing values to return the final index:
        GroupMat   = np.zeros((nobs))
        GroupMat[:]   = np.nan
        # ready to assign:
        GroupMat[ind_s] = ind_optimal

        # ready to save:
        self.groups  = GroupMat
        self.WithVar = WithVar[min_maxvar]        
    
    
    #--------------------------------------------------
    # METHODS OF THE KMEANS CLASS        
    #--------------------------------------------------
    def kmeans_means(self):
        ''' For each group produced above, get a matrix with means 
            and std.errs.  Here I use Pandas capabilities
        '''
        # Things I need
        Xmat  = self.Xmat_ori  # possibly with missing values
        nvars = self.nvars
        K     = self.K
        N     = self.N
        # Get clusters 
        groups = pd.DataFrame(self.groups)
        # concatenate
        names  = ['var_'+str(v) for v in range(nvars)]
        names1 = np.concatenate((['Groups'],names),axis=1)
        cl_ind = ['Cluster ' + str(k) for k in range(K)]
        BigX = pd.DataFrame(np.concatenate((groups,Xmat),axis=1),
                            columns=names1)
        #exclude first column from next calculations
        x_cols = BigX.columns[1:]
        #        means    = BigX[x_cols].groupby(BigX[0]).mean()
        # get means and standard deviations
#        means    = np.asarray(BigX[x_cols].groupby(BigX[0]).mean())
#        stds     = np.asarray(BigX[x_cols].groupby(BigX[0]).std())
        num_to_labs = {g:cl_ind[g] for g in range(K)}    
        BigX['Groups']  = BigX['Groups'].map(num_to_labs)
        means    = BigX.groupby(BigX['Groups']).mean()
        stds     = BigX[x_cols].groupby(BigX['Groups']).std(ddof=0)/np.sqrt(N)
        nobs_grp = BigX[x_cols].groupby(BigX['Groups']).count()
        nobs_grp = nobs_grp.iloc[:,0]
        return means,stds, nobs_grp
        
    def kmeans_means_ols(self):
        ''' For each group produced above, get a matrix with means 
            and std.errs
            But instead of using Pandas I want to run regressions
            This guarantees that means and std. errors are well computed
        '''
        # Things I need
        Xmat  = self.xmat  # possibly with missing values
        N     = self.N
        nvars = self.nvars
        K     = self.K
        ind_n = self.ind
        # Get clusters 
        groups = self.groups[ind_n]
        predum    = pd.DataFrame(groups)
        dummies   = np.asarray(pd.get_dummies(predum[0]))
        nobs_grp  = dummies.sum(axis=0)
        # Iterate and run regressions
        MeanMat  = np.zeros((K,nvars))
        StdMat   = np.zeros((K,nvars))
        StdMat1  = np.zeros((K,nvars))
        for v in range(nvars):
            yy   = Xmat[:,v].reshape((N,1))
            xx   = np.concatenate((np.ones((N,1)),dummies[:,1:]),axis=1)
            ols_mn = ols.ols_dan(yy,xx)
            premeans = np.tile(ols_mn.betahat()[0],((K),1))
            premeans[0] = 0
            means  = ols_mn.betahat() + premeans 
            # get standard error: for reference: vcv[0,0]
            # for remaing ones: vcv[0,0] + vcv[k,k] + vcv[0,k]
            term0 = np.tile(ols_mn.vcv_white()[0,0],(K,1))
            term1 = np.diag(ols_mn.vcv_white()).copy().reshape((K,1))
            term1[0] = 0
            term2 = ols_mn.vcv_white()[0,:].copy().reshape((K,1))
            term2[0] = 0
            var_groups = term0 + term1 + 2*term2 
            stds  = np.sqrt(var_groups)
            # check with regression with all dummies
            ols_mn = ols.ols_dan(yy,dummies)
            stds1  = ols_mn.stderr_sph()
            # ready to save
            MeanMat[:,v] = means.flatten()
            StdMat[:,v]  = stds.flatten()
            StdMat1[:,v] = stds1.flatten()
        return MeanMat,StdMat,nobs_grp

    def kmeans_anova(self):
        ''' 
        I want two matrices as output:
        DGrp, DAll = kmeans_anova()
        ----------------------------
        
        1. DGrp is a (GxGxNvars) matrix
            For each var I have a GxG matrix
            For row g \in G: D_{g,k,var}= 1 if H0: \beta_{g,k}^i=0 can be rej 
                             at 5% when g is reference (i.e. means are 
                            different)
            Regression is like this:
            x_{i,g} = \alpha_g + \sum_{k \neq g} \beta_{g,k}^i D_k + \epsilon_{i,g}
            
            Therefore, if I get the mean across vars for each group g it
            tells me the fraction of other groups that are different from g
            in explaining all variables
            
        2. DAll is a (Nvars x 1) vector:
            For each var I test equality of coefficients.
            D_{var} = 1 if H0:all means are equal is rejected (ie. some 
                        group is different)
        '''
        # Things I need
        Xmat  = self.xmat  # possibly with missing values
        N     = self.N
        nvars = self.nvars
        K     = self.K
        ind_n = self.ind
        # Get clusters 
        groups = self.groups[ind_n]
        predum    = pd.get_dummies(pd.DataFrame(groups)[0])
        DiffMatGroups = np.zeros((K,K,nvars))
        DiffMatGroups[:,:,:] = np.nan
        DiffMatAll    = np.zeros((nvars,1))
        for v in range(nvars):
            yy = Xmat[:,v].reshape((N,1))
            #----------------------------------------------------
            # First regression: check if all groups are the same:
            #----------------------------------------------------
            xx = np.concatenate((np.ones((N,1)),np.asarray(predum)[:,1:]),axis=1)
            ols_unc = ols.ols_dan(yy,xx)
            R2_unc  = ols_unc.R2()
            xx = np.ones((N,1))
            ols_con = ols.ols_dan(yy,xx)
            R2_con  = ols_con.R2()
            fstat   = ((R2_unc - R2_con/(K-1))/((1-R2_unc)/(N-nvars)))
            fstatpv = 1 - stats.f.cdf(fstat,(K-1), (N-nvars))
            # H0: all means are equal--> if reject at 5%, then some groups are diff
            if fstatpv<0.05:
                DiffMatAll[v] = 1
            #----------------------------------------------------
            # Second: loop through all groups changing the reference
            #----------------------------------------------------
            for g in np.arange(K):
                cols = np.setdiff1d(np.arange(K),[g])
                dums = predum[cols]
                xx  = np.concatenate((np.ones((N,1)),dums),axis=1)
                ols_grp = ols.ols_dan(yy,xx)
                # if pval < 1% reject null hyp of betas equal to zero(= ref.grp)
                chk_sig = ols_grp.pval_white()[1:]<0.01
                DiffMatGroups[g,cols,v] = chk_sig.flatten()
                
        return DiffMatGroups, DiffMatAll
        
        
        
    def gap_statistic(self):
        '''
        Compute the Gap Statistic to determine number of clusters
        Reference: Tibshirani, et.al (2001) or Hastie, et.al. p. 519
        Definition: Gap = E[LogW] - Log(WithVar)
        I need: 
        1. E[LogW]: 
            a. Get B uniform random samples over the box delimited
                by [min x_k, max x_k] for each k= 1:nvars
            b. Given the clusters I have estimated computed LogW as above
                but with the fake data
            3. Compute sample average of LogW
        2. Log(WithVar): from above
        #------------------------------------------
        Note: this is not working because for each g \in G that I'm estimating 
              from outside changes the 
        '''
        Xmat  = self.xmat  # possibly without missing values
        N     = self.N
        nvars = self.nvars
        K     = self.K
        ind_n = self.ind
        d     = self.distance
        # Get clusters 
        groups = self.groups[ind_n]
        # Get what I already have from before: WithinVar
        LogWithVar = np.log(self.WithinVar)
        # Bootstrapsamples        
        BB = 100
        # I need to get max and mins of all vars: BxNxNvars
        minx = np.tile(np.tile(np.nanmin(Xmat, axis=0),(N,1)),(BB,1,1))
        maxx = np.tile(np.tile(np.nanmax(Xmat, axis=0),(N,1)),(BB,1,1))
        # Draw uniform random variables
        U = np.random.rand(BB,N,nvars)
        XMatBoot = minx + (maxx-minx)*U
        # loop over Bs
        WithVarB = np.zeros((BB,1))
        for b in range(BB):
            Xboot    = XMatBoot[b,:,:]
            Xmat_pos = pd.DataFrame(np.concatenate((groups,Xboot),axis=1))
            for g in range(K):
                # use only those in group g:
                X_g = np.asarray(Xmat_pos[Xmat_pos[0]==g])
                X_g = X_g[:,1:]
                # I need distances for each var across i,i' \in g
                DMat = np.zeros((X_g.shape[0],X_g.shape[0],nvars))
                for v in range(nvars):
                    # compute D(i,i'), for (i,i') \in g
                    if d==1:
                        DMat[:,:,v] = (X_g[:,v].reshape((X_g.shape[0],1))
                                 - X_g[:,v].reshape((1,X_g.shape[0])))**2
                    elif d==2:
                        DMat[:,:,v] = np.abs(X_g[:,v].reshape((X_g.shape[0],1))
                                 - X_g[:,v].reshape((1,X_g.shape[0])))
                # ready to compute within variance
                # Notice I'm already adding up over clusters
                if d==1:
                    WithVarB[b] += 0.5*np.sqrt(DMat.sum(axis=2).sum())
                elif d==2:
                    WithVarB[b] += 0.5*(DMat.max(axis=2).sum())
        # Done with loop, get logs and average
        ExpLogWith  = np.log(WithVarB).mean()
        # Gap can be readily estimated
        Gap = ExpLogWith - LogWithVar
        return ExpLogWith, LogWithVar