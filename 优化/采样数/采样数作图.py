import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'

x1=np.loadtxt('采样数10_0.txt')
inter=10/x1[:,0]
tau1=x1[:,1:10001]/100

x2=np.loadtxt('采样数10_0.1.txt')
tau2=x2[:,1:10001]/100

x3=np.loadtxt('采样数10_0.5.txt')
#print(x3[:,0])
#inter3=10/x1[7:,0]
tau3=x3[:,1:10001]/100
tau3=np.sort(tau3)
tau4=np.clip(tau3,0,10)

plt.figure(dpi=150)
plt.plot(inter,np.ones(len(inter)),'k')

plt.plot(inter,np.mean(tau1,axis=1),'--',color=c1,lw=7,
         label=r'$\sigma_e=0$')
#plt.plot(inter,np.mean(tau1,axis=1)+np.std(tau1,axis=1),'--',color=c1)
#plt.plot(inter,np.mean(tau1,axis=1)-np.std(tau1,axis=1),'--',color=c1)
plt.plot(inter,np.percentile(tau1,16,axis=1),'--',color=c1)
plt.plot(inter,np.percentile(tau1,84,axis=1),'--',color=c1)

plt.plot(inter,np.mean(tau2,axis=1),color=c3,lw=4,
         label=r'$\sigma_{\rm e}\simeq0.1\sigma$')
plt.fill_between(inter,np.percentile(tau2,16,axis=1),
                 np.percentile(tau2,84,axis=1),
                 color=c3,alpha=0.3,edgecolor="none")
#plt.plot(inter,np.mean(tau2,axis=1)+np.std(tau2,axis=1),'--',color=c2)
#plt.plot(inter,np.mean(tau2,axis=1)-np.std(tau2,axis=1),'--',color=c2)
#plt.plot(inter,np.percentile(tau2,16,axis=1),'--',color=c2)
#plt.plot(inter,np.percentile(tau2,84,axis=1),'--',color=c2)

plt.plot(inter,np.mean(tau4,axis=1),'-.',color=c2,lw=3,
         label=r'$\sigma_{\rm e}\simeq0.5\sigma$')
plt.plot(inter,np.percentile(tau3,16,axis=1),'-.',color=c2)
plt.plot(inter,np.percentile(tau3,84,axis=1),'-.',color=c2)
#plt.plot(inter,np.mean(tau3,axis=1)+np.std(tau3,axis=1),'--',color=c3)
#plt.plot(inter,np.mean(tau3,axis=1)-np.std(tau3,axis=1),'--',color=c3)

plt.xscale('log')
plt.xlabel(r'$<\Delta t>/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.title(r'$T=10\tau_{\rm in}$',fontsize=20)
plt.grid()
plt.legend(loc=4,fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.minorticks_on()
plt.xlim(1,0.01)
plt.ylim(0,1.6)
plt.show()

