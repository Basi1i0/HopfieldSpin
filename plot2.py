# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:59:09 2019

@author: Basil
"""

def Y(y,alpha):
    return np.erf(y)/(np.sqrt(2*alpha)+2/np.sqrt(np.pi)*np.exp(-y**2))



plt.rcParams["figure.figsize"] = [8, 8]
y=np.linspace(0,4.2)
plt.plot(y, Y(y, 0.03), color='black')
plt.plot(y, Y(y, 0.06), color='darkblue')
plt.plot(y, Y(y, 0.138), color='blue')
plt.plot(y, Y(y, 0.3), color='royalblue')
plt.plot(y,y, '--', color='red')
plt.xlabel('$y$')
plt.legend(['$\\alpha=0.03$', '$\\alpha=0.06$', '$\\alpha=0.138$', '$\\alpha=0.3$'])

plt.ylim([0,4.2])
plt.xlim([0,4.2])

y=np.linspace(0,2)
plt.plot(y, Y(y, 0.03), color='black')
plt.plot(y, Y(y, 0.06), color='darkblue')
plt.plot(y, Y(y, 0.138), color='blue')
plt.plot(y, Y(y, 0.3), color='royalblue')
plt.plot(y,y, '--', color='red')
plt.ylim([0,2])
plt.xlim([0,2])
plt.xlabel('$y$')
plt.legend(['$\\alpha=0.03$', '$\\alpha=0.06$', '$\\alpha=0.138$', '$\\alpha=0.3$'])