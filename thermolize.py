# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:22:27 2019

@author: Basil
"""
import numpy as np;

def Thermalize(Sinit, J, T, max_iter, max_cycle = 0,  change_tol = 0): 
    
    if(max_cycle == 0): max_cycle = max_iter;
    
    Nspins = Sinit.shape[0];
    fliper = np.ones(Nspins)-2*np.identity(Nspins)
    
    Scurrent = np.copy(Sinit)
    Sall = np.zeros(shape = (max_iter, Nspins))

    unchanged = 0
    
    for i in range(max_iter):
        Sall[i,:] = Scurrent
        
        E0 = np.matmul(Scurrent, np.matmul(J, Scurrent))
        ms = np.zeros(Nspins)
        
        for s in range(Nspins):
            E1 = np.matmul(Scurrent*fliper[s,:], np.matmul(J, Scurrent*fliper[s,:]));
   
            if(T == 0):
                ms[s] = np.sign((E0 - E1))
            else:
                p = np.exp(-(E0 - E1)/Nspins/T)
                if(p > 1): p = 1
                ms[s] = np.random.choice([-1,1], p=[p, 1-p])

        Scurrent = Scurrent*ms
        
        if( sum(ms == -1) <= change_tol*Nspins  ): unchanged = unchanged + 1;
        if( unchanged >= max_cycle ): break
    
    #Niterations_performed = i;
    return (Sall[0:i,:])