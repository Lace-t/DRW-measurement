import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#aa2116'
c5='#ffe600'

x=np.loadtxt('de60j2.txt')
for i in range(len(x)):
    t=x[i,0]
    x[i,1:]=x[i,1:]/t
    x[i,0]=2922/t

MAX=np.maximum(x[:,4],x[:,1])

plt.figure(dpi=150)
plt.plot(x[:,0],np.ones(len(x[:,0])),'k')

plt.plot(x[:,0],x[:,1],color=c2,lw=3,label=r'$\tau_{\rm MLE}$',alpha=0.5)
plt.plot(x[:,0],x[:,4],color=c3,lw=3,label=r'$\tau_{\rm PM}$',alpha=0.5)
plt.plot(x[:,0],MAX,'--',color=c5,lw=9,alpha=0.5,label=r'$\tau_{\rm MAX}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}/\tau_{\rm in}$',fontsize=20)
plt.title(r'K17 prior(N=60,$\sigma_{\rm e}=\sigma_{\rm SDSS}$)',fontsize=20)
#plt.title(r'K17 prior(N=445,$\sigma_{\rm e}=\sigma_{\rm OGLE}$)',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
