import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#aa2116'
c5='#ffe600'

tin=10*np.logspace(0,2,41)
x1=np.loadtxt('c12gap0v.txt')
tau1=np.sqrt(x1[:,1:10001])/0.1414
#x2=np.loadtxt('c6gap6.txt')
#tau2=x2[:,1:10001]/tin.reshape(41,1)
x3=np.loadtxt('c12gap9v.txt')
tau3=np.sqrt(x3[:,1:10001])/0.1414

plt.figure(dpi=150)
plt.plot(x1[:,0],np.ones(len(x1)),'k')

plt.plot(x1[:,0],np.mean(tau1,axis=1),color=c1,label='12 month')
plt.plot(x1[:,0],np.percentile(tau1,16,axis=1),'--',color=c1)
plt.plot(x1[:,0],np.percentile(tau1,84,axis=1),'--',color=c1)

#plt.plot(x1[:,0],np.mean(tau2,axis=1),color=c2,label='6 month(30 epochs/yr)')
#plt.plot(x1[:,0],np.percentile(tau2,16,axis=1),'--',color=c2)
#plt.plot(x1[:,0],np.percentile(tau2,84,axis=1),'--',color=c2)

plt.plot(x1[:,0],np.mean(tau3,axis=1),color=c3,label='3 month')
plt.plot(x1[:,0],np.percentile(tau3,16,axis=1),'--',color=c3)
plt.plot(x1[:,0],np.percentile(tau3,84,axis=1),'--',color=c3)

plt.xscale('log')
plt.xlabel(r'$\tau_{\rm in}/d$',fontsize=20)
plt.ylabel(r'$\sigma_{\rm out}^{\rm mean}/\sigma_{\rm in}$',fontsize=20)
plt.title(r'<cadence> = 12 d,$\sigma_e=\sigma_{\rm SDSS}$',fontsize=20)

plt.xlim(1000,10)
plt.grid()
plt.ylim(0.5,1.2)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

