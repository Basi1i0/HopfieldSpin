# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:14:32 2019

@author: Basil
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os
from mpl_toolkits.mplot3d import Axes3D


def ReadResults(path):
    all_filenames = [f for f in os.listdir(path) if f.endswith('.pickled')];
    results = list();
    
    for i in range(len(all_filenames)):
        filename = path + os.listdir(path)[i]
        filehandlerread = open(filename, 'rb');
        computation_result = pickle.load(filehandlerread)
        results.append(computation_result)
    return results;
       
def MsDistribution(results):    
    final_overlaps = np.zeros( ( len(results), results[0]['P']) )   
    for k in range( len(results)):
        for i in range(results[k]['P']):
            final_overlaps[k,i] = np.matmul(results[k]['xi'][i,:], results[k]["Sall"][-1,:]) / results[k]["Nspins"]
        final_overlaps[k,:] = final_overlaps[k, np.argsort(np.abs(final_overlaps[k,:]))]
    return final_overlaps;

def QsDistribution(results):
    qs = np.empty( len(results) ) 
    for i in range(len(results)):
        qs[i] = np.matmul(results[i]['Sall'][-1,:], results[i]['Sall'][-1,:])/results[i]['Nspins']
    
    return qs
#    qs = np.empty((len(results), len(results)))
#    qs[:] = np.nan
#    for i in range(len(results)):
#        for j in range(len(results)):
#            if(i == j): continue
#            qs[i,j] = np.matmul(results[i]['Sall'][-1,:], results[j]['Sall'][-1,:])/results[i]['Nspins'];
#    return qs[~np.isnan(qs)]  

def RsDistribution(results):
    ms = MsDistribution(results)
    rs = np.sum(ms[:,range(ms.shape[1] - 1)]**2, 1)/results[0]["alpha"]
    return rs
#    rs = np.empty((len(results), len(results)))
#    rs[:] = np.nan
#    for i in range(len(results)):
#        for j in range(len(results)):
#            if(i == j): continue
#            rs[i,j] = np.matmul(np.abs(ms[i, range(ms.shape[1] - 1)]), np.abs(ms[j, range(ms.shape[1] - 1)])) / results[i]['alpha'];
#    return rs[~np.isnan(rs)]

def Plot2dDistributionEvolution(x, ys, pointdunction = np.median, noise_level = 0):
    for i in range(len(ys)):
        plt.plot([x[i] + (np.random.rand(1) - 0.5)*noise_level for _ in range(len(ys[i]))], ys[i],
                 'o', mfc='none',color='black', markersize = 5)#, showmedians = True, widths = 1)
    plt.plot(x + (np.random.rand(len(x)) - 0.5)*noise_level, list(map(pointdunction, ys)), 
             '--',color='blue', linewidth=4 )#, showmedians = True, widths = 1)

def Plot3dDistributionEvolution(x, ys, r, nbins = 20 ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(ys)):
        h = np.histogram(ys[i], bins = nbins, range = r)
    
        bins = (h[1][1:] + h[1][:-1])/2
        counts = h[0]/sum(h[0])
    
        ax.bar(bins, counts, zs=x[i], zdir='y', color=plt.get_cmap('jet')(i/len(ys)), 
               alpha=0.8, width =  np.diff(h[1]), edgecolor  = 'white')
    return ax
   
#    plt.show()

def EsDistribution(results):
    Es = np.empty(len(results));
    for i in range(len(results)):
        J=(np.matmul(np.transpose(results[i]["xi"]), results[i]["xi"]) - np.identity(results[i]["Nspins"]) )/results[i]["Nspins"]
        Es[i] = -1/2*(np.matmul(results[i]["Sall"][-1,:], np.matmul(J, results[i]["Sall"][-1,:])))/results[i]["Nspins"]
        
    return Es
        



basepath = "C:\\Users\\Basil\\ResearchData\\Hopfield\\T=0.0002_in=0.36\\" 
folders = os.listdir(basepath);
m0s = list();
qsa = list();
rsa = list();
Esa = list();
alphas = list();
for i in range(len(folders)):
    path = basepath +folders[i]+ "\\"
    results = ReadResults(path)
    ms = MsDistribution(results)
    m0s.append(np.abs(ms[:,-1]))
    
    qs = QsDistribution(results)
    qsa.append(qs)
    
    rs = RsDistribution(results)
    rsa.append(rs)
    
    Es = EsDistribution(results)
    Esa.append(Es)
    
    alphas.append(results[0]["alpha"])

plt.rcParams["figure.figsize"] = [12/2.5, 8/2.5]
   

plt.rcParams.update({'font.size': 14})
Plot2dDistributionEvolution(alphas, m0s, np.mean, noise_level = 0.0002)
plt.xlabel('$\\alpha$')
#plt.xlim(min(alphas)*0.95, max(alphas)*1.05)
#plt.xlim(0.03, 0.14)
plt.ylim(0.2, 1.05)
plt.ylabel('$|m^{0}|$')


ax = Plot3dDistributionEvolution(alphas[1:-4], m0s[1:-5], [0.3, 1], 30)
ax.set_xlabel('$|m^{0}|$')
ax.set_ylabel('$\\alpha$')
#ax.set_zlabel('')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(30,290)