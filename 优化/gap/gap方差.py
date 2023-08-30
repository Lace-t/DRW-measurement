import numpy as np
import emcee
import celerite
import DRW_library as dl
from celerite import terms
from scipy.optimize import minimize
from multiprocessing import Pool
from time import perf_counter
start=perf_counter()

def obs():
    t=np.array([])
    epochs=round((365-30*gap)/cadence)
    for i in range(10):
        t=np.append(t,np.sort(np.random.randint(0,365-30*gap,epochs)+
                              np.random.uniform(-0.13,0.13,epochs))+365*i)
    return t    

def mainprocess(m):
    tau=10*1.122**m
    re=np.array([])
    for k in range(N):       
        np.random.seed(seed[m*N+k])
        t=obs()
        y=dl.DRW_process(t,tau,0.2,17)
        ls_sdss=[]
        lsigma=[]
        for i in range(len(t)):
            s=np.sqrt(0.004**2+np.exp(1.63*(y[i]-22.55)))
            #s=np.sqrt(0.013**2+np.exp(2*(y[i]-23.36)))
            lsigma.append(s)
            ls_sdss.append(np.random.normal(y[i],s,1)[0])
        s_sdss=np.array(ls_sdss)
        sigma=np.array(lsigma)

        re=np.var(s_sdss)*len(s_sdss)/(len(s_sdss)-1)-np.mean(np.power(sigma,2))
    print('[',m,']',np.log10(2922/tau))
    return np.concatenate(([tau],re))
    
#BEGIN
cadence=6
N=10000
gap=0
seed=np.random.randint(1,2**31-1,41*N)
print('cadence=',cadence,'gap=',gap,'N=',N)
if __name__ == '__main__':
    pool=Pool(41)
    l=np.array(pool.map(mainprocess, range(41)))
    pool.close()
    pool.join()    
np.savetxt('c6gap0v.txt',l,fmt='%.4f')
print('time=',perf_counter()-start)
#END    
