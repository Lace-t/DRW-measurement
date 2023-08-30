import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#aa2116'
c5='#ffe600'

'''
x=np.loadtxt('de445j2.txt')
for i in range(len(x)):
    t=x[i,0]
    x[i,1:]=x[i,1:]/t
    x[i,0]=2922/t

plt.figure(dpi=150)
plt.plot(x[:,0],np.ones(len(x[:,0])),'k')

plt.plot(x[:,0],x[:,3],'--',color=c1,lw=5,label=r'$\tau_{\rm PE}$')
plt.plot(x[:,0],x[:,4],color=c3,lw=5,label=r'$\tau_{\rm PM}$')
plt.plot(x[:,0],x[:,2],'-.',color=c4,lw=5,label=r'$\tau_{\rm MAP}$')
plt.plot(x[:,0],x[:,1],':',color=c2,lw=5,label=r'$\tau_{\rm MLE}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.text(10,0.1,'mean',fontsize=20)
#plt.title(r'K17 prior(N=60,$\sigma_e=\sigma_{\rm SDSS}$)',fontsize=20)
plt.title(r'K17 prior(N=445,$\sigma_{\rm e}=\sigma_{\rm OGLE}$)',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(dpi=150)
plt.plot(x[:,0],np.ones(len(x[:,0])),'k')

plt.plot(x[:,0],x[:,7],'--',color=c1,lw=5,label=r'$\tau_{\rm PE}$')
plt.plot(x[:,0],x[:,8],color=c3,lw=5,label=r'$\tau_{\rm PM}$')
plt.plot(x[:,0],x[:,6],'-.',color=c4,lw=5,label=r'$\tau_{\rm MAP}$')
plt.plot(x[:,0],x[:,5],':',color=c2,lw=5,label=r'$\tau_{\rm MLE}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm median}/\tau_{\rm in}$',fontsize=20)
plt.text(10,0.1,'median',fontsize=20)
#plt.title(r'K17 prior(N=60,$\sigma_e=\sigma_{\rm SDSS}$)',fontsize=20)
plt.title(r'K17 prior(N=445,$\sigma_{\rm e}=\sigma_{\rm OGLE}$)',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
'''

rho=np.logspace(3,0,61)
tin=np.logspace(0.4657,3.4657,61)
x=np.load('de445j2.npy')
tau=x[:,:,0:4]/tin.reshape(61,1,1)
sigma=x[:,:,4:]/0.14

plt.figure(dpi=150)
plt.plot(rho,np.ones(len(rho)),'k')

plt.fill_between(rho,np.percentile(tau,16,axis=1)[:,3],np.percentile(tau,84,axis=1)[:,3],
                 color=c3,alpha=0.3,edgecolor="none")


plt.plot(rho,np.mean(tau,axis=1)[:,2],'--',color=c1,lw=5,label=r'$\tau_{\rm PE}$')
plt.plot(rho,np.mean(tau,axis=1)[:,3],color=c3,lw=5,label=r'$\tau_{\rm PM}$ (K17PMm)')
plt.plot(rho,np.mean(tau,axis=1)[:,1],'-.',color=c4,lw=3,label=r'$\tau_{\rm MAP}$')
plt.plot(rho,np.mean(tau,axis=1)[:,0],':',color=c2,lw=3,label=r'$\tau_{\rm MLE}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.text(10,0.2,'mean',fontsize=20)
#plt.title(r'K17 prior (N = 60, $\sigma_e=\sigma_{\rm SDSS}$)',fontsize=20)
plt.title(r'K17 prior (N = 445, $\sigma_{\rm e}=\sigma_{\rm OGLE}$)',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0.1,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure(dpi=150)
plt.plot(rho,np.ones(len(rho)),'k')

plt.fill_between(rho,np.percentile(tau,16,axis=1)[:,2],np.percentile(tau,84,axis=1)[:,2],
                 color=c1,alpha=0.3,edgecolor="none")

plt.plot(rho,np.median(tau,axis=1)[:,2],'--',color=c1,lw=5,label=r'$\tau_{\rm PE}$ (S21PE50)')
plt.plot(rho,np.median(tau,axis=1)[:,3],color=c3,lw=5,label=r'$\tau_{\rm PM}$')
plt.plot(rho,np.median(tau,axis=1)[:,1],'-.',color=c4,lw=3,label=r'$\tau_{\rm MAP}$')
plt.plot(rho,np.median(tau,axis=1)[:,0],':',color=c2,lw=3,label=r'$\tau_{\rm MLE}$')

plt.xscale('log')
plt.xlabel(r'$T/\tau_{\rm in}$',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm median}/\tau_{\rm in}$',fontsize=20)
plt.text(10,0.2,'median',fontsize=20)
#plt.title(r'K17 prior (N = 60, $\sigma_e=\sigma_{\rm SDSS}$)',fontsize=20)
plt.title(r'S21 prior (N = 445, $\sigma_{\rm e}=\sigma_{\rm OGLE}$)',fontsize=20)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0.1,1.5)
plt.legend(fontsize=15,loc=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
