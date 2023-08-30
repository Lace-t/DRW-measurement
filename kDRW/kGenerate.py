import numpy as np
from time import perf_counter
start=perf_counter()

def DRW_process(t,tau,SF,m):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

N=1000
epochs=445
print('epochs=',epochs,'N=',N)
ln=[]
for i in np.arange(0,151,4):
    tau=2922/1.047**i
    print(i)
    for j in range(N):
        t=np.sort(np.random.randint(0,2922,epochs)+
                  np.random.uniform(-0.13,0.13,epochs))
        y=DRW_process(t,tau,0.2,17)
        ls_sdss=[]
        lsigma=[]
        for k in range(epochs):
            s=np.sqrt(0.004**2+np.exp(1.63*(y[k]-22.55)))
            #s=np.sqrt(0.013**2+np.exp(2*(y[k]-23.36)))
            lsigma.append(s)
            ls_sdss.append(np.random.normal(y[k],s,1)[0])
        s=np.array(ls_sdss)
        sigma=np.array(lsigma)
        ln.append('lc'+str(i)+'_'+str(j))
        np.savetxt(ln[-1],np.c_[t,s,sigma],fmt='%f')

f=open('process.dat','w')
f.write(str(N*38)+'\n')
for i in ln:
    f.write('445 '+i+'\n')
f.close()
print(perf_counter()-start)
