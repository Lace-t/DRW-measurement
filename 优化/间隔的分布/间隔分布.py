import numpy as np
import DRW_library as dl
from multiprocessing import Pool
from time import perf_counter
start=perf_counter()

def mainprocessing(m):
    tau=2.922*1.122**m
    re=np.array([])
    for k in range(N):       
        np.random.seed(seed[m*N+k])
        #t=np.linspace(0,2922,epochs)
        t=np.sort(np.random.randint(0,2922,epochs)+
                  np.random.uniform(-0.13,0.13,epochs))
        y=dl.DRW_process(t,tau,0.2,17)
        
        ls_sdss=[]
        lsigma=[]
        for i in range(epochs):
            s=np.sqrt(0.004**2+np.exp(1.63*(y[i]-22.55)))
            #s=np.sqrt(0.013**2+np.exp(2*(y[i]-23.36)))
            lsigma.append(s)
            ls_sdss.append(np.random.normal(y[i],s,1)[0])
        s_sdss=np.array(ls_sdss)
        sigma=np.array(lsigma)
        try:
            re=np.append(re,DRW_fit(t,s_sdss,sigma,np.mean(s_sdss)))
        except:
            continue
    re.resize((N,8))  
    print('[',m,']',np.log10(2922/tau))
    return re

#BEGIN
N=1000
seed=np.random.randint(1,2**31-1,101*N)
print('N=',N)
if __name__ == '__main__':
    pool=Pool(41)
    l=list(pool.map(mainprocessing, range(41)))
    pool.close()
    pool.join()
np.savetxt('random.txt',np.array(l))
print('time=',time.perf_counter()-start)
#END
