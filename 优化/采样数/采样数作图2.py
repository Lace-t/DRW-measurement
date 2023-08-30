import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#6f599c'

x2=np.loadtxt('采样数10_0.1.txt')
inter=10/x2[:,0]
tau2=x2[:,1:10001]/100

x4=np.loadtxt('采样数100_0.1.txt')
tau4=x4[:,1:10001]/100

plt.figure(dpi=150)
plt.plot(inter,np.ones(len(inter)),'k')

plt.plot(inter,np.mean(tau2,axis=1),color=c3,lw=5,
         label=r'$T=10\tau_{\rm in}$')
plt.fill_between(inter,np.percentile(tau2,16,axis=1),
                 np.percentile(tau2,84,axis=1),
                 color=c3,alpha=0.3,edgecolor="none")
#plt.plot(inter,np.percentile(tau2,16,axis=1),'--',color=c2)
#plt.plot(inter,np.percentile(tau2,84,axis=1),'--',color=c2)

plt.plot(inter,np.mean(tau4,axis=1),'--',color=c4,lw=3,
         label=r'$T=100\tau_{\rm in}$')
plt.plot(inter,np.percentile(tau4,16,axis=1),'--',color=c4)
plt.plot(inter,np.percentile(tau4,84,axis=1),'--',color=c4)


plt.xscale('log')
plt.xlabel(r'$<\Delta t>/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.title(r'$\sigma_{\rm e}\simeq0.1\sigma$',fontsize=20)
#plt.title(r'$\sigma=0.14$ mag,$\sigma_e=0.1\sigma$',fontsize=20)
plt.grid()
plt.legend(loc=4,fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.minorticks_on()
plt.xlim(1,0.01)
plt.ylim(0,1.6)
plt.show()

