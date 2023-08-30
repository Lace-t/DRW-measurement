import numpy as np
from multiprocessing import Pool
import math
import time
import DRW_library as dl
import celerite
from celerite import terms
from scipy.optimize import minimize
start=time.perf_counter()

def obs(cad):
    if cad in [5,10]:
        t=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        for i in range(9):
            t1=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
        return t
    elif cad==30:
        t=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
        return t
    elif cad==60:
        t=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
        t0=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
        t=np.append(t,t0)
        return np.sort(t)

def mainprocessing(m):
    tau=m
    at=np.array([])
    for k in range(N):
        np.random.seed(seed[round(m/20*N+k)])
       
        #模拟在某天的不同时刻观测
        t1=obs(cad)
        t=np.array([])
        for i in range(len(t1)):
            if t1[i]< 365.25*3:
                t=np.append(t,t1[i])
            else:
                break       
        y=dl.DRW_process(t,tau,0.2,18)
        e=np.full_like(y,0.03)
        s=np.random.normal(y,e)
        try:
            re=dl.DRW_fit(t,s,e)
        except:
            continue
        at=np.append(at,re[0]/tau)
    print(tau,np.mean(at),np.std(at))
    return [tau,np.mean(at),np.std(at)]

#BEGIN
N=1000
cad=5
seed=np.random.randint(0,2**31-1,10**6)
print('N=',N,'cadence=',cad,'epochs/month')
if __name__ == '__main__':
    pool=Pool(31)
    l=list(pool.map(mainprocessing,np.linspace(1,600,31)))
    pool.close()
    pool.join()    
np.savetxt('10c5t3.txt',np.array(l))
print('time=',time.perf_counter()-start)
#END
