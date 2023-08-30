import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#aa2116'
c5='#ffe600'

rho=np.logspace(0,3,61)

x=np.loadtxt('cadence0.2.txt')
tau=x[:,1:10001]/100

x2=np.loadtxt('cadence1.txt')
tau2=x2[:,1:10001]/100
#print(np.mean(tau2,axis=1))

plt.figure(dpi=150)
plt.plot(rho,np.ones(len(rho)),'k')

plt.fill_between(rho,np.percentile(tau,16,axis=1),np.percentile(tau,84,axis=1),
                 color=c3,alpha=0.3,edgecolor="none")
plt.plot(rho,np.mean(tau,axis=1),color=c3,lw=3,
         label=r'$<\Delta t>=0.2\tau_{\rm in}$')

plt.plot(rho[21:],np.mean(tau2,axis=1)[21:],'--',lw=3,color=c2,
         label=r'$<\Delta t>=1\tau_{\rm in}$')
plt.plot(rho,np.percentile(tau2,16,axis=1),'--',color=c2)
plt.plot(rho,np.percentile(tau2,84,axis=1),'--',color=c2)


plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.title(r'$\sigma_e\simeq0.1\sigma$',fontsize=20)
plt.xlim(10,1000)
plt.grid()
plt.ylim(0.4,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.minorticks_on()
plt.show()
