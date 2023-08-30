import numpy as np
import time
import DRW_library as dl
import celerite
from celerite import terms
from scipy.optimize import minimize
from multiprocessing import Pool
start=time.perf_counter()

def SPZ(name):
    f=open(root+'QSO_S82/'+name)
    l=[]
    for line in f:
            line=line.replace('\n','')
            l.append(list(map(eval, line.split(" "))))
    f.close()
    lt_sdss=[]
    le_sdss=[]
    lm_sdss=[]
    for i in range(len(l)):
        lt_sdss.append(dl.MJDtoCE(l[i][6]))
        if l[i][8]<0.02:
            le_sdss.append(l[i][8]+0.01)
        else:
            le_sdss.append(l[i][8])
        lm_sdss.append(l[i][7])

    lmc=lm_sdss.copy()
    lec=le_sdss.copy()
    ltc=lt_sdss.copy()
    for i in range(len(lmc)):
        if lmc[i]>25 or lmc[i]<15:
            lm_sdss.remove(lmc[i])
            lt_sdss.remove(ltc[i])
            le_sdss.remove(lec[i])
        
    t_sdss=np.array(lt_sdss)
    sigma_sdss=np.array(le_sdss)
    #PS1
    f=open(root+'PS1/'+name)
    l=[]
    for line in f:
        line=line.replace('\n','')
        l.append(list(map(eval, line.split(" ")[:-1])))
    f.close()
    lt_ps=[]
    le_ps=[]
    lm_ps=[]
    for i in range(len(l)):
        lt_ps.append(dl.MJDtoCE(l[i][0]))
        if l[i][2]<0.02:
            le_ps.append(l[i][2]+0.01)
        else:
            le_ps.append(l[i][2])
        lm_ps.append(l[i][1])
    lmc=lm_ps.copy()
    lec=le_ps.copy()
    ltc=lt_ps.copy()
    for i in range(len(lmc)):
        if lmc[i]>25 or lmc[i]<15:
            lm_ps.remove(lmc[i])
            lt_ps.remove(ltc[i])
            le_ps.remove(lec[i])
    #去除坏点
    m_sdss=0
    m_ps=0
    for i in lm_sdss:
        m_sdss+=i
    for i in lm_ps:
        m_ps+=i
    mean=(m_sdss+m_ps)/(len(lm_sdss)+len(lm_ps))

    lec=le_ps.copy()
    ltc=lt_ps.copy()
    lmc=lm_ps.copy()
    for i in range(len(lmc)):
        if abs(lmc[i]-mean)>7*lec[i]:
            le_ps.remove(lec[i])
            lt_ps.remove(ltc[i])
            lm_ps.remove(lmc[i])
    t_ps=np.sort(np.array(lt_ps))
    sigma_ps=np.array(le_ps)
    #ZTF
    d_ztf=np.loadtxt(root+'ZTF/'+name+'.txt')
    t_ztf=np.array(list(map(dl.MJDtoCE,list(d_ztf[:,0]))))
    sigma_ztf=d_ztf[:,2]
    
    t=np.sort((np.concatenate((t_sdss,t_ps,t_ztf))-1998)*365.25)
    sigma=np.concatenate((sigma_sdss,sigma_ps,sigma_ztf))
    return t,sigma,mean

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
        name=str(round(ln[k]))
        t0,e,mean=SPZ(name)
        np.random.seed(round(m//40)*N+k)
        t1=obs(cad)+24*365.25
        t2=np.concatenate((t0,t1))
        t=np.array([])
        for i in range(len(t2)):
            if t2[i]<365.25*34:
                t=np.append(t,t2[i])
        e=np.concatenate((e,np.full(len(t)-len(t0),si)))
        y=dl.DRW_process(t,tau,0.2,mean)
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
si=0.03
cad=5
root='/data3/lace/MyLibrary/Simulation/'
seed=np.random.randint(0,2**31-1,41*N*10)
ln=np.loadtxt(root+'content_ZTF.txt')
print('N=',N,'sigma=',si,'Cadence=',cad,'/month')
if __name__ == '__main__':
    pool=Pool(41)
    l=list(pool.map(mainprocessing,np.linspace(1,1200,41)))
    pool.close()
    pool.join()    
np.savetxt('SPZ34c5t10.txt',np.array(l),fmt='%.4f')
print('time=',time.perf_counter()-start)
#END
