import numpy as np
from multiprocessing import Pool
import DRW_library as dl
import time
import celerite
from celerite import terms
from scipy.optimize import minimize
start=time.perf_counter()

def obs(period):
    cad=30
    if period==1:
        t=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        return t
    else:
        t=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        for i in range(5):
            t1=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
        return t

def mainprocessing(m):
    tau=m
    at=np.array([])
    for k in range(N):
        np.random.seed(seed[round(N*m+k)])
        #模拟在某天的不同时刻观测
        t=obs(cad)
        y=dl.DRW_process(t,tau,0.2,18)
        e=np.full_like(y,sigma)
        s=np.random.normal(y,e)
        try:
            re=dl.DRW_fit(t,s,e)
            at=np.append(at,re[0]/tau)
        except:
            print(tau,'Err')
            continue
    print(tau,np.mean(at))
    return np.concatenate(([tau],at))

#BEGIN
N=10
period=1
sigma=0.014
seed=np.random.randint(0,2**31-1,N*100)
print('N=',N,'sigma=',sigma,'period=',period)
if __name__ == '__main__':
    pool=Pool(10)
    l=list(pool.map(mainprocessing, np.arange(1,11)))
    pool.close()
    pool.join()    
np.savetxt('c1t1e1.txt',np.array(l))
print('time=',time.perf_counter()-start)
#END
