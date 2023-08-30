import numpy as np
import matplotlib.pyplot as plt
import random 

def DRW_process(t,tau,SF,m):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def obs(cad):
    if cad in [5,10,30]:
        t=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        for i in range(9):
            t1=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
        return t
    elif cad==60:
        t=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
        for i in range(9):
            t1=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
            
        for i in range(10):
            t1=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
            t0=i*365.25+t1
            t=np.append(t,t0)    
        return np.sort(t)

t0=np.arange(0,3652)
y=DRW_process(t0,200,0.2,18)
t=obs(5)
e=np.full_like(t,0.03)
s=np.array([])
for ti in t:
    s=np.append(s,np.random.normal(y[round(ti)-1],0.03))

plt.figure(figsize=(15,6))
plt.plot(t0/365.25+2023,y,'k',lw=0.2,label='Simulated Light Curve')
plt.errorbar(t/365.25+2023,s,yerr=e,fmt='.r',label='WFST')
plt.xlabel('t/yr')
plt.ylabel('m/mag')
plt.title(r'WFST($\tau=200d,\sigma=0.14\ mag,m=18\ mag$,baseline=10yr,15epochs/yr,Error=0.03)')
plt.xlim(2023,2033)
plt.legend()
plt.show()

j=0
for i in range(len(t)):
    if i!=len(t)-1:
        if t[i+1]>j*365.25:
            print('\n第{0:d}年：'.format(j+1))
            j=j+1
    print(int(t[i]),end=',')
