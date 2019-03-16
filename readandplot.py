# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:14:32 2019

@author: Basil
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os


def ReadResults(path):
    all_filenames = os.listdir(path);
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
    np.empty( len(results) ) 
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

def PlotDistributionEvolution(x, ys, pointdunction = np.median, figsize = [9, 6]):
    plt.rcParams["figure.figsize"] = figsize
    for i in range(len(ys)):
        plt.plot([x[i] for _ in range(len(ys[i]))], ys[i], '.b')#, showmedians = True, widths = 1)
    plt.plot(x, list(map(pointdunction, ys)), 'r', linewidth=2 )#, showmedians = True, widths = 1)

def EsDistribution(results):
    Es = np.empty(len(results));
    for i in range(len(results)):
        J=(np.matmul(np.transpose(results[i]["xi"]), results[i]["xi"]))/results[i]["Nspins"]
        Es[i] = (np.matmul(results[i]["Sall"][-1,:], np.matmul(J, results[i]["Sall"][-1,:])))/results[i]["Nspins"]
        
    return Es
        



basepath = "C:\\Users\\Basil\\ResearchData\\Hopfield\\T=0.0002_in=0.5\\" 
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
    
    #Es = EsDistribution(results)
    #Esa.append(Es)
    
    alphas.append(results[0]["alpha"])



PlotDistributionEvolution(alphas, m0s, np.median)
plt.plot(alphas,(np.sqrt(2/np.pi/np.array(alphas)))**2 )