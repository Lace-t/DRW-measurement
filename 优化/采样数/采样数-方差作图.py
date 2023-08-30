import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#6f599c'

x2=np.loadtxt('采样数10v.txt')
inter=10/x2[:,0]
tau2=np.sqrt(x2[:,1:10001])/0.1414

x4=np.loadtxt('采样数100v.txt')
tau4=np.sqrt(x4[:,1:10001])/0.1414

plt.figure(dpi=150)


plt.plot(inter,np.mean(tau2,axis=1),color=c2,
         label=r'$T=10\tau_{\rm in}$')
#plt.plot(inter,np.percentile(tau2,16,axis=1),'--',color=c2)
#plt.plot(inter,np.percentile(tau2,84,axis=1),'--',color=c2)

plt.plot(inter,np.mean(tau4,axis=1),color=c4,
         label=r'$T=100\tau_{\rm in}$')
#plt.plot(inter,np.percentile(tau4,16,axis=1),'--',color=c4)
#plt.plot(inter,np.percentile(tau4,84,axis=1),'--',color=c4)


plt.xscale('log')
plt.xlabel(r'$<\Delta t>/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\sigma_{\rm out}^{\rm mean}/\sigma_{\rm in}$',fontsize=20)
plt.title(r'$\sigma_e=0.1\sigma$',fontsize=20)
plt.grid()
plt.legend(loc=2,fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(1,0.1)
plt.ylim(0.8,1.2)
plt.show()

