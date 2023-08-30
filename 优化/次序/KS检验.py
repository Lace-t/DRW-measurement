import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import matplotlib.gridspec as gd

ai=np.loadtxt('Initial.txt')/200
ac=np.loadtxt('Changed.txt')/200
au=np.loadtxt('Uniform.txt')/200

plt.figure(dpi=150)
plt.hist(ai[:,0],bins=np.linspace(0.95,1,30),facecolor='g',histtype='step',label='Initial')
plt.hist(ac[:,0],bins=np.linspace(0.95,1,30),facecolor='b',histtype='step',label='Changed')
plt.hist(au[:,0],bins=np.linspace(0.95,1,30),facecolor='r',histtype='step',label='Uniform')
plt.legend()
plt.xlabel(r'$\tau_{out}/\tau_{in}$')
plt.ylabel('N')
plt.title(r'$\tau=200d,\sigma=0.14,baseline=15\tau,N=100,Error=0$')
plt.show()

plt.figure(dpi=150)
gs=gd.GridSpec(2,2)
ax1=plt.subplot(gs[0,0])
ax2=plt.subplot(gs[0,1])
ax3=plt.subplot(gs[1,0])
ax4=plt.subplot(gs[1,1])
ax1.hist(ai[:,0],bins=np.linspace(0.95,1,20),facecolor='g',histtype='step',label='Initial')
ax1.set_ylabel('N')
ax1.legend(fontsize=8,loc=2)

ax2.hist(ac[:,0],bins=np.linspace(0.95,1,20),facecolor='b',histtype='step',label='Changed')
ax2.legend(fontsize=8,loc=2)

ax3.hist(au[:,0],bins=np.linspace(0.95,1,20),facecolor='r',histtype='step',label='Uniform')
ax3.set_ylabel('N')
ax3.set_xlabel(r'$\tau_{out}/\tau_{in}$')
ax3.legend(fontsize=8,loc=2)

ax4.hist(np.abs(ai[:,0]-ac[:,0]),facecolor='g',histtype='step',label='|I-C|')
ax4.hist(np.abs(ai[:,0]-au[:,0]),facecolor='b',histtype='step',label='|I-U|')
ax4.set_xlabel(r'$\tau_{out}/\tau_{in}$')
ax4.legend(fontsize=8)
#plt.title(r'$\tau=200d,\sigma=0.14,baseline=15\tau,N=100,Error=0$')
plt.show()

plt.figure(dpi=150)
plt.hist(np.abs(ai[:,0]-ac[:,0]),facecolor='g',histtype='step',label='|I-C|')
plt.hist(np.abs(ai[:,0]-au[:,0]),facecolor='b',histtype='step',label='|I-U|')
plt.xlabel(r'$\tau_{out}/\tau_{in}$')
plt.ylabel('N')
plt.title(r'$\tau=200d,\sigma=0.14,baseline=15\tau,N=100,Error=0$')
plt.legend()
plt.show()

print(kstest(ac[:,0],ai[:,0]))
print(kstest(ac[:,0],au[:,0]))
