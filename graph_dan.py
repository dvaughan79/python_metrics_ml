# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:58:18 2015
This module will be used to set up new graphs that I need to customize
@author: a3940004
"""
import pandas as pd
import numpy  as np
import scipy.io
import olsdan as ols
import missing    as mi
import multicol   as mcol
import matplotlib
import matplotlib.pyplot as plt

class graph_dan: 
    '''
    Date: 15-04-2015
    Customize some graphs that I need
    '''
    def __init__(self,*args):
        # before defining attributes clean data
        nargs   = len(args)
        if nargs == 2: # I only included y,X
            Xmat  = args[0]
            Ymat  = args[1]
            order = 1
        elif nargs == 3:
            Xmat  = args[0]
            Ymat  = args[1]
            order = args[2]
            names = np.asarray(['X'+str(i) for i in range(Xmat.shape[1])])
        elif nargs == 4:
            Xmat  = args[0]
            Ymat  = args[1]
            order = args[2]
            names = args[3]
        # get everything as an array:
        nobs = Ymat.shape[0]
        #nvar = Xmat.shape[0]
        y    = np.asarray(Ymat).reshape((nobs,1))
        X    = np.asarray(Xmat).reshape((nobs,1))
        # ready to get these:
        self.y  = y
        self.X  = X
        self.names = names
        self.order = order
        
    # This is my main function
    def plot_linreg_scatter(self):
        X = self.X
        y = self.y
        names = self.names
        order = self.order
        # ready to start
        plt.scatter(X,y, color='b', alpha=0.5, edgecolor='k', s=12)
        # get labels and title
        plt.xlabel(names[0],fontsize=12)
        plt.ylabel(names[1],fontsize=12)
        tit_lab = u'Relación Entre ' + names[1] + ' y ' + names[0]
        plt.title(tit_lab)
        nobs = y.shape[0]
        # fix y
        y = np.asarray(y).reshape((nobs,1))
        # 2. Run a regression that depends on order:
        xrlin = np.asarray(X).reshape((nobs,1))
        xr2   = xrlin**2
        xr3   = xrlin**3
        xr4   = xrlin**4
        xr5   = xrlin**5
        onesr = np.ones((nobs,1))
        # now for fit
        xlin = np.linspace(pd.Series(X).min(),pd.Series(X).max(),100).reshape((100,1))
        x2   = xlin**2
        x3   = xlin**3
        x4   = xlin**4
        x5   = xlin**5
        onesv = np.ones((100,1))
        # Assemble matrices:
        if order ==1:
            xmat    = np.concatenate((onesr,xrlin),axis=1)
            xmatlin = np.concatenate((onesv,xlin),axis=1)
        elif order ==2:
            xmat    = np.concatenate((onesr,xrlin, xr2),axis=1)
            xmatlin = np.concatenate((onesv,xlin, x2),axis=1)
        elif order ==3:
            xmat    = np.concatenate((onesr,xrlin,xr2,xr3),axis=1)
            xmatlin = np.concatenate((onesv,xlin,x2,x3),axis=1)
        elif order ==4:
            xmat    = np.concatenate((onesr,xrlin,xr2,xr3,xr4),axis=1)
            xmatlin = np.concatenate((onesv,xlin,x2,x3,x4),axis=1)
        else :
            xmat    = np.concatenate((onesr,xrlin,xr2,xr3,xr4,xr5),axis=1)
            xmatlin = np.concatenate((onesv,xlin,x2,x3,x4,x5),axis=1)
        # ready to run ols:
        olsg = ols.ols_dan(y,xmat)
        #
        ylin = np.dot(xmatlin, np.asarray(olsg.betahat()).reshape((order+1,1)))
        # plot
        plt.plot(xlin,ylin, color='r', alpha=0.5)
        plt.show()