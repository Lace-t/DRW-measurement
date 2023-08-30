import numpy as np
import DRW_library as dl
from time import perf_counter
from multiprocessing import Pool
start=perf_counter()

def mainprocessing(m):
    re=np.array([])
    for k in range(N):
        epochs=round(10*1.2589**m)
        np.random.seed(seed[m*N+k])
        #t=np.linspace(0,2922,epochs)
        t=np.sort(np.random.randint(0,1000,epochs)+
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
            re=np.append(re,dl.DRW_fit(t,s_sdss,sigma))
        except:
            continue
    re.resize((N,2))
    re=re.T.flatten()
    print('[',epochs,']')
    return np.concatenate(([epochs],re))

#BEGIN
N=10000
tau=100
seed=np.random.randint(1,2**31-1,21*N)
print('N=',N)
if __name__ == '__main__':
    pool=Pool(21)
    l=list(pool.map(mainprocessing, range(21)))
    pool.close()
    pool.join()
np.savetxt('采样数10_0.1.txt',np.array(l),fmt='%.4f')
print('time=',perf_counter()-start)
#END
