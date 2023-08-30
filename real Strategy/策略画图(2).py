import numpy as np
import matplotlib.pyplot as plt

a=np.loadtxt('c1t1e1.txt')
rt=a[:,1:]
rt=np.sort(rt)

b=np.loadtxt('c1t6e1.txt')

c=np.loadtxt('c1t10e1.txt')

d=np.loadtxt('t0.txt')
rt3=d[:,1:]
rt3=np.sort(rt3)

c1="#009ad6"
c3='#6f599c'
c2='#84bf96'
c4='#b54334'

plt.figure(dpi=150)
plt.plot(a[:,0],np.ones(len(a[:,0])),'k')
#plt.plot(a[:,0],np.full(len(a[:,0]),0.95),'k:')

plt.plot(a[:,0],np.mean(rt3[:,10:-10],axis=1),'-.',color=c1,lw=3,label=r'SDSS+PS1')
plt.plot(a[:,0],np.percentile(d[:,1:],16,axis=1),'-.',color=c1)
plt.plot(a[:,0],np.percentile(d[:,1:],84,axis=1),'-.',color=c1)

plt.plot(a[:,0],np.mean(rt[:,5:-5],axis=1),':',color=c4,lw=3,label=r'+W1')
plt.plot(a[:,0],np.percentile(rt,16,axis=1),':',color=c4)
plt.plot(a[:,0],np.percentile(rt,84,axis=1),':',color=c4)

plt.plot(b[:,0],np.mean(b[:,1:],axis=1),color=c2,lw=3,
         label=r'+W6')
plt.fill_between(c[:,0],np.percentile(b[:,1:],16,axis=1),
                 np.percentile(b[:,1:],84,axis=1),
                 color=c2,alpha=0.3,edgecolor="none")

plt.plot(c[:,0],np.mean(c[:,1:],axis=1),'--',color=c3,lw=3,
         label=r'+W10')
plt.plot(c[:,0],np.percentile(c[:,1:],16,axis=1),'--',color=c3)
plt.plot(c[:,0],np.percentile(c[:,1:],84,axis=1),'--',color=c3)


plt.ylim(0.25,1.5)
plt.xlim(10,1000)
plt.grid()
plt.xlabel(r'$\tau_{\rm in}$/d',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.title(r'$\sigma=0.14$ mag, $\sigma_{\rm W}\simeq0.1\sigma$, $\Delta t_W=1$ d',
          fontsize=20)
plt.xticks((10,200,400,600,800,1000),fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=1,fontsize=15)
plt.minorticks_on()
plt.show()
