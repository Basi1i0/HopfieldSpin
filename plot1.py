# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:40:37 2019

@author: Basil
"""


plt.rcParams.update({'font.size': 14})
plt.xlabel('$\\alpha$')
Plot2dDistributionEvolution(alphas, [np.add(x, 0.03) for x in Esa], np.median, noise_level = 0.0002)
plt.xlim(0.125, 0.16)
#plt.ylim(-0.55, -0.35)
plt.ylabel('$H$')
plt.plot(alphas,-1/np.pi - 1*np.sqrt(2*np.array(alphas)/np.pi) , 'r' )
#plt.plot(alphas,[-1/2]*len(alphas), 'r' )


ax = Plot3dDistributionEvolution(alphas[1:-4], [np.add(x, 0.03) for x in Esa[1:-4]], [-0.66, -0.52], 30)
ax.set_xlabel('$H$')
ax.set_ylabel('$\\alpha$')
#ax.set_zlabel('')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(30,290)
ax.set_zticklabels([])




plt.rcParams["figure.figsize"] = [12, 8]
   
plt.rcParams.update({'font.size': 14})
plt.xlabel('$\\alpha$')
Plot2dDistributionEvolution(alphas, m0s, np.median, noise_level = 0.0002)
#plt.xlim(min(alphas)*0.95, max(alphas)*1.05)
plt.xlim(0.125, 0.16)
plt.ylabel('$|m^{0}|$')


ax = Plot3dDistributionEvolution(alphas[1:-4], m0s[1:-5], [0.3, 1], 30)
ax.set_xlabel('$|m^{0}|$')
ax.set_ylabel('$\\alpha$')
#ax.set_zlabel('')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(30,290)
ax.set_zticklabels([])


results = ReadResults("C:\\Users\\Basil\\ResearchData\\Hopfield\\T=0.0002_in=0.5\\a=0.03\\");
xl = [0,171]

Plot(results[3])
plt.xlim(xl)
plt.ylim([-1,1])
plt.xlabel('iteration')
plt.ylabel('$m^{\\mu}$ overlap')

plt.rcParams["figure.figsize"] = [8, 8*2/3]
PlotE(results[3])
plt.xlim(xl)
plt.ylim([-0.7,0])
plt.xlabel('iteration')
plt.ylabel('$H$')