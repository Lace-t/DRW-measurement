import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c2='#f15a22'
c3='#7fb80e'
c4='#6f599c'
n='u'

x=np.loadtxt('j445.txt')
x0=np.loadtxt('S-60.txt')
x2=np.loadtxt('f60.txt')
x3=np.loadtxt('f60U.txt')
x4=np.loadtxt('k445.txt')
x5=np.loadtxt('k60p.txt')

for i in range(2):
    x[:,i+1]=x[:,i+1]/x[:,0]
x[:,0]=2922/x[:,0]

for i in range(2):
    x0[:,i+1]=x0[:,i+1]/x0[:,0]
x0[:,0]=2922/x0[:,0]

for i in range(2):
    x2[:,i+1]=x2[:,i+1]/x2[:,0]
x2[:,0]=2922/x2[:,0]

for i in range(2):
    x3[:,i+1]=x3[:,i+1]/x3[:,0]
x3[:,0]=2922/x3[:,0]

for i in range(3):
    x4[:,i+1]=x4[:,i+1]/x4[:,0]
x4[:,0]=2922/x4[:,0]

for i in range(2):
    x5[:,i+1]=x5[:,i+1]/x5[:,0]
x5[:,0]=2922/x5[:,0]

plt.figure(dpi=150)
plt.plot(x[:,0],np.ones(len(x[:,0])),'k')
if n=='j':
    plt.plot(x[:,0],x[:,1],color=c4,label=r'$mean(JAVELIN)$')
    plt.plot(x[:,0],x[:,1]+x[:,2],'--',color=c4,label=r'$\sigma_G(JAVELIN)$')
    plt.plot(x[:,0],x[:,1]-x[:,2],'--',color=c4)
elif n=='s':
    plt.plot(x0[:,0],x0[:,1],color=c1,label=r'mean(S21)')
    plt.plot(x0[:,0],x0[:,1]+x0[:,2],'--',color=c1,label=r'$\sigma_G$(S21)')
    plt.plot(x0[:,0],x0[:,1]-x0[:,2],'--',color=c1)
elif n=='u':
    plt.plot(x3[:,0],x3[:,1],color=c1,label=r'mean(Uniform)')
    plt.plot(x3[:,0],x3[:,1]+x3[:,2],'--',color=c1,label=r'$\sigma_G$(Uniform)')
    plt.plot(x3[:,0],x3[:,1]-x3[:,2],'--',color=c1)
elif n=='k':
    plt.plot(x4[:,0],x4[:,1],':',color=c3,label=r'median(K17)')
    plt.plot(x4[:,0],x4[:,2],'--',color=c3,label=r'$\sigma_G$(K17)')
    plt.plot(x4[:,0],x4[:,3],'--',color=c3)
elif n=='kp':
    plt.plot(x5[:,0],x5[:,1],color=c3,label=r'mean(K17)')
    plt.plot(x5[:,0],x5[:,1]+x5[:,2],'--',color=c3,label=r'$\sigma_G$(K17)')
    plt.plot(x5[:,0],x5[:,1]-x5[:,2],'--',color=c3)
plt.plot(x2[:,0],x2[:,1],color=c2,lw=3,label=r'mean(Our work)')
plt.fill_between(x2[:,0],x2[:,1]+x2[:,2],x2[:,1]-x2[:,2],color=c2,alpha=0.3,
                 edgecolor="none",label=r'$\sigma_G(Our work)$')
#plt.plot(x2[:,0],x2[:,1]+x2[:,2],'--',color=c2,label=r'$\sigma_G$(Our work)')
#plt.plot(x2[:,0],x2[:,1]-x2[:,2],'--',color=c2)

#plt.plot(x0[:,0],x0[:,1],color=c1)
plt.xscale('log')
plt.xlabel(r'$T/\tau_{in}$',fontsize=15)
plt.ylabel(r'$\tau_{out}/\tau_{in}$',fontsize=15)
#plt.title(r"N=60,$\sigma_{\rm e}=\sigma_{\rm SDSS}$",fontsize=20)
plt.title(r"N=445,$\sigma_{\rm e}=\sigma_{\rm OGLE}$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(1,1000)
plt.grid()
plt.ylim(0,2)
plt.legend(fontsize=15,loc=4)
plt.show()
