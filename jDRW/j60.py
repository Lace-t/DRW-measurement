import numpy as np
from javelin.lcmodel import Cont_Model
from javelin.zylc import get_data
from multiprocessing import Pool

def DRW_process(t,tau,SF,m):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def jDRW_fit(t,s,err,m):
    dname='d'+str(m)+'.txt'
    fname='f'+str(m)+'.dat'
    np.savetxt(dname,np.c_[t,s,err])
    x=get_data(dname)
    cont = Cont_Model(x)
    cont.do_mcmc(set_prior=True,nwalkers=10,nburn=50,nchain=200,fchain=fname)
    re=np.loadtxt(fname)
    r=np.median(re,axis=0)
    return np.exp(np.array([r[1],r[0]]))

def mainprocessing(m):
    tau=2922/1.047**m
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
            #s=np.sqrt(0.004**2+np.exp(1.63*(y[i]-22.55)))
            s=np.sqrt(0.013**2+np.exp(2*(y[i]-23.36)))
            lsigma.append(s)
            ls_sdss.append(np.random.normal(y[i],s,1)[0])
            
        s_sdss=np.array(ls_sdss)
        sigma=np.array(lsigma)
        re=np.append(re,jDRW_fit(t,s_sdss,sigma,m))
    re.resize((N,2))
    print('[',m,']',np.log10(2922/tau))
    return np.array([tau,np.mean(re[:,0]),np.std(re[:,0])])
    
#BEGIN
epochs=60
N=1000
seed=np.random.randint(1,2**31-1,200*N)
print('epochs=',epochs,'N=',N)
if __name__ == '__main__':
    pool=Pool(38)
    l=np.array(pool.map(mainprocessing,np.arange(0,151,4)))#
    pool.close()
    pool.join()    
np.savetxt('j60.txt',l,fmt='%.4f')
#END    
