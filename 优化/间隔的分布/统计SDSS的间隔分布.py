import numpy as np
import matplotlib.pyplot as plt
import os
import numba
'''
def SDSS(name):
    f=open(root+'/'+name)
    l=[]
    for line in f:
            line=line.replace('\n','')
            l.append(list(map(eval, line.split(" "))))
    f.close()
    lt_sdss=[]
    for i in range(len(l)):
        lt_sdss.append(l[i][6])
    t_sdss=np.array(lt_sdss)
    return t_sdss

root='/home/charon/桌面/AGN光学光变的模拟和测量/Simulation/QSO_S82'
files = os.listdir(root)

dt=np.array([])
for i in files:
    t=SDSS(str(i))
    dt=np.concatenate((dt,np.diff(t)))
print(len(dt))
np.savetxt('dt.txt',dt)
'''
dt=np.loadtxt('dt.txt')
dt=dt[dt[:]<180]
print(np.mean(dt))
plt.hist(dt,bins=180,range=(0,180))
plt.xlim(0,180)
plt.show()


