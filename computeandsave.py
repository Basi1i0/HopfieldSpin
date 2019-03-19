import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os, glob
import timeit

from joblib import Parallel, delayed
#from thermolize import *


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

def Compute(Nspins, Niterations, alpha, T, initial_noise, seed, max_cycle = 20, change_tol = 0):
    np.random.seed(seed)
    
    
    P = int(round(alpha*Nspins))
    
    xi = np.array([(np.random.randint(2,size = Nspins)*2 - 1) for _ in range(P)])
    J=(np.matmul(np.transpose(xi), xi))/Nspins
    
    initial_ind = 0;
    Sinit = xi[initial_ind,:]*np.random.choice([1,-1], size = Nspins, replace = True, 
                                               p = [1-initial_noise, initial_noise])#
    
    Sall = Thermalize(Sinit, J, T, Niterations, 20, 0.003)
    
    return {'Nspins': Nspins,  'alpha' : alpha, 'T' : T, 
            'initial_noise' : initial_noise, 'seed' : seed,
            'Sall' : Sall, 'xi' : xi, 'P' : P, 
            'Niterations': Niterations, 'max_cycle' : max_cycle, 'change_tol' : change_tol}

def Plot(computed):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.hist(np.array([np.matmul(computed['xi'][i,:], computed['Sall'][0,:]) for i in range(computed['P'])])/computed['Nspins'])
    plt.subplot(1, 2, 2)
    plt.hist(np.array([np.matmul(computed['xi'][i,:], computed['Sall'][-1,:]) for i in range(computed['P'])])/computed['Nspins'])
    
    init_overlaps = np.array([np.matmul(computed['xi'][i,:], computed['Sall'][0,:]) for i in range(computed['P'])])/computed['Nspins']
    final_overlaps = np.array([np.matmul(computed['xi'][i,:], computed['Sall'][-1,:]) for i in range(computed['P'])])/computed['Nspins']
    #np.argmax(abs(overlaps))
    final_ind =  np.argmax(abs(init_overlaps))#initial_ind;
    
    plt.figure(2)
    for i in range(computed['P']):
        dSall = np.array([np.matmul( computed['xi'][i,:], computed['Sall'][j,:])/Nspins for j in range(computed['Sall'].shape[0]) ])
        col = 'b'
        if(i == final_ind): col = 'r'
        plt.plot(dSall, col)   
        

def PlotE(computed):
    J = (np.matmul(np.transpose(computed['xi']), computed['xi']) - 
                   np.identity(computed["Nspins"]))/computed["Nspins"]
    
    plt.figure(2)
    dSall = np.array([0.5*np.matmul(-computed['Sall'][j,:],
                          np.matmul(J, computed['Sall'][j,:]))/Nspins for j in range(computed['Sall'].shape[0]) ])
    col = 'b'
    plt.plot(dSall, col, linewidth = 4)   

def SampleAndSave(Nsamples, Nspins, Niterations, alpha, T, initial_noise, path):
    os.makedirs(path, exist_ok = True)
    
    start = 0
    
    existing_files = [f for f in os.listdir(path) if f.endswith('.pickled')]
    existing = [int(os.path.splitext(file)[0]) for file in existing_files]

    if(len(existing) > 0): start = max(existing) + 1
    
    for i in range(start, Nsamples + start): 
        computed = Compute(Nspins, Niterations, alpha, T, initial_noise, i)
        
        filename = path + str(i) + '.pickled'
        
        filehandler = open(filename, 'wb');
        pickle.dump(computed, filehandler)
        filehandler.close()
 


Nspins = 1000
Niterations = 200
#alpha = 0.14
T = 0.0002
initial_noise = 0.05
basepath = "C:\\Users\\Basil\\ResearchData\\Hopfield\\T="+ str(T) +"_in=" + str(initial_noise) + "\\" 

nsteps = 10

alphas = np.linspace(0.03, 0.039, num= nsteps )
paths = [basepath + "a=" +str(np.round(alphas[i],6)) + "\\" for i in range(nsteps)]

print(basepath)
print(alphas)

num_cores = 3

if __name__ == '__main__':
    start_time = timeit.default_timer()
    Parallel(n_jobs=num_cores)(delayed(SampleAndSave)(20, Nspins, Niterations, alphas[i], T, initial_noise, paths[i]) for i in range(nsteps))
    elapsed = timeit.default_timer() - start_time


   
#computation_result =  Compute(Nspins, Niterations,  0.03, 0.0002, 0.5, np.random.randint(100)) #pickle.load(filehandlerread)
#
#Plot(computation_result)
#plt.xlabel('iteration')
#plt.ylabel('$m^{\\mu}$ overlap')



