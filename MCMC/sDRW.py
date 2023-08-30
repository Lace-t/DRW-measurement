import numpy as np
import Suberlak_module as mod
from multiprocessing import Pool
from time import perf_counter
start=perf_counter()

def DRW_process(t,tau,SF,m):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    #ls=[m]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def sDRW_fit(t,s,err,prior='Jeff1'): 
    # fit the observed simulated LC and real LC 
    # with Celerite MAP and Expectation value ...
    
    # set limits for grid, MAP , and prior 
    sig_lims , tau_lims = [0.02,0.7], [1,5000]#边界很有意思

    # find the Celerite expectation value and MAP
    # find the MAP estimate
    
    sigmaMAP,tauMAP,gp=mod.find_celerite_MAP(t,s,err,prior=prior,set_bounds=True,
                       sig_lims=sig_lims,tau_lims=tau_lims ,verbose=False)
    
    # expectation from grid  
    Ngrid = 60;scale='log'
    sigma_grid,tau_grid=mod.make_grid(scale,sig_lims,tau_lims,Ngrid) 
    logP=mod.evaluate_logP(sigma_grid,tau_grid,s,gp,prior,'celerite')

    # find the expectation value 
    sigmaEXP,tauEXP=mod.find_expectation_value(logP,sigma_grid,tau_grid)
    return np.array([tauEXP,sigmaEXP])

prior='Jeff1' #  because this is 1/sigma * 1/tau ...

def mainprocessing(m):
    tau=2.922*1.122**m
    re=np.array([])
    for k in range(N):       
        np.random.seed(seed[m*N+k])
        #t=np.linspace(0,2922,epochs)
        t=np.sort(np.random.randint(0,2922,epochs)+
                  np.random.uniform(-0.13,0.13,epochs))
        y=DRW_process(t,tau,0.2,18)
        ls_sdss=[]
        lsigma=[]
        for i in range(epochs):
            s=np.sqrt(0.004**2+np.exp(1.63*(y[i]-22.55)))
            #s=np.sqrt(0.013**2+np.exp(2*(y[i]-23.36)))
            lsigma.append(s)
            ls_sdss.append(np.random.normal(y[i],s,1)[0])
            
        s_sdss=np.array(ls_sdss)
        sigma=np.array(lsigma)
        re=np.append(re,sDRW_fit(t,s_sdss,sigma,prior))
    re.resize((N,2))
    re=re.T.flatten()
    print('[',m,']',np.log10(2922/tau))
    return np.concatenate(([tau],re))
    
#BEGIN
epochs=445
N=10000
seed=np.random.randint(1,2**31-1,200*N)
print('epochs=',epochs,'N=',N)
if __name__ == '__main__':
    pool=Pool(61)
    l=np.array(pool.map(mainprocessing, range(61)))
    pool.close()
    pool.join()    
np.savetxt('S445.txt',l,fmt='%.4f')
print('time=',perf_counter()-start)
#END    
