import numpy as np
import matplotlib.pyplot as plt
c1="#009ad6"
c3='#6f599c'
c2='#84bf96'

b=np.loadtxt('1t6e1.txt')
bm=[]
b16=[]
b84=[]
for i in b:
    i=i[1:]
    j=np.sort(i[i[:]>0])
    #print(len(j))
    bm.append(np.mean(j))
    b16.append(np.percentile(j,16))
    b84.append(np.percentile(j,84))
    
c=np.loadtxt('1t6e5.txt')
cm=[]
c16=[]
c84=[]
for i in c:
    i=i[1:]
    j=np.sort(i[i[:]>0])
    print(len(j))
    cm.append(np.mean(j))
    c16.append(np.percentile(j,16))
    c84.append(np.percentile(j,84))

plt.figure(dpi=150)
plt.plot(b[:,0],np.ones(len(b[:,0])),'k')


plt.plot(b[:,0],bm,color=c2,lw=3,
         label=r'$\sigma_{\rm W}\simeq0.1\sigma$')
#plt.plot(b[:,0],np.percentile(b[:,1:],16,axis=1),'--',color=c2)
#plt.plot(b[:,0],np.percentile(b[:,1:],84,axis=1),'--',color=c2)
plt.fill_between(b[:,0],b16,b84,color=c2,alpha=0.3,edgecolor="none")

plt.plot(c[:,0],cm,'--',color=c3,lw=3,
         label=r'$\sigma_{\rm W}\simeq0.5\sigma$')
plt.plot(c[:,0],c16,'--',color=c3)
plt.plot(c[:,0],c84,'--',color=c3)
plt.ylim(0.1,1.9)
plt.xlim(0.1,10)
plt.xscale('log')
plt.grid()
plt.xlabel(r'$\tau_{\rm in}$/d',fontsize=20)
plt.ylabel(r'$\tau_{\rm out}^{\rm mean}/\tau_{\rm in}$',fontsize=20)
plt.title(r'T = 6 yr, $\sigma=0.14$ mag, $\Delta t_W=1$ d',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=1,fontsize=15)
plt.minorticks_on()
plt.show()
