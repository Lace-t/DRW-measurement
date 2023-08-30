import numpy as np
import os
import matplotlib.pyplot as plt
'''
tau=3
t=np.sort(np.random.uniform(0,2922,60))
y=dl.DRW_process(t,tau,0.2,18)
e=np.full_like(y,0.03)
np.savetxt('jt.dat',np.c_[t,y,e])
'''
for i in np.arange(0,151,4):
    d='d'+str(i)+'.txt'
    f='f'+str(i)+'.dat'
    try:
        os.remove(d)
    except:
        pass
    try:
        os.remove(f)
    except:
        pass
'''
x=np.loadtxt('j60.txt')
plt.figure(dpi=150)
plt.plot(x[:,0]/2922,x[:,0]/2922,'k',label='y=x',alpha=0.5)
plt.plot(x[:,0]/2922,x[:,1]/2922,'r--',label='median')
plt.plot(x[:,0]/2922,(x[:,1]+x[:,2])/2922,'r',label=r'$\sigma_G$')
plt.plot(x[:,0]/2922,(x[:,1]-x[:,2])/2922,'r')
plt.xlim(0.001,1)
plt.ylim(0.001,1)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
#ax = plt.gca()
#ax.invert_yaxis()
#ax.invert_xaxis()
plt.xlabel(r'$lg(\tau_{in}/T)$')
plt.ylabel(r'$lg(\tau_{out}/T)$')
#plt.title('Suberlak N=60m')
plt.show()

x=np.loadtxt('ft.dat')
re=np.mean(x,axis=0)
print(np.exp(re))
plt.hist(x[:,1],bins=50)
plt.show()
'''
