import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#aa2116'
c5='#ffe600'

rho=np.logspace(3,0,61)
tin=np.logspace(0.4657,3.4657,61)

x=np.load('de445j2.npy')
sigma=x[:,:,4:]/0.1414
x2=np.load('de445j1.npy')
sigma2=x[:,:,4:]/0.1414

y=np.sqrt(1-2/rho+2/rho**2*(1-np.exp(-rho)))

plt.figure(dpi=150)
plt.plot(rho,np.ones(len(rho)),'k')

plt.fill_between(rho,np.percentile(sigma,16,axis=1)[:,3],
                 np.percentile(sigma,84,axis=1)[:,3],
                 color=c3,alpha=0.3,edgecolor="none")

plt.plot(rho,np.median(sigma2,axis=1)[:,2],'--',color=c1,lw=4,label=r'S21PE50')
plt.plot(rho,np.mean(sigma,axis=1)[:,3],color=c3,lw=4,label=r'K17PMm')
plt.plot(rho,np.mean(sigma,axis=1)[:,0],color=c2,lw=2,label=r'MLE')
plt.plot(rho,y,'k:',lw=2,label=r'$\sqrt{1-\frac{2}{x}+\frac{2}{x^2}(1-e^{-x})}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\sigma_{\rm out}/\sigma_{\rm in}$',fontsize=20)
#plt.title(r'K17 prior (N = 60, $\sigma_e=\sigma_{\rm SDSS}$)',fontsize=20)
plt.title(r'N = 445, $\sigma_{\rm e}=\sigma_{\rm OGLE}$',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0.3,1.2)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


