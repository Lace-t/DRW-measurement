import numpy as np
from multiprocessing import Pool
import time
import DRW_library as dl
import celerite
from celerite import terms
from scipy.optimize import minimize
start=time.perf_counter()

def SP(name):
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
    t=np.sort((np.concatenate((t_sdss,t_ps))-1998)*365.25)
    sigma=np.concatenate((sigma_sdss,sigma_ps))
    return t,sigma,mean

def obs(cad):
    epochs=int(90/cad)
    t=np.linspace(1,91,epochs+1)[:-1]+np.random.uniform(-0.13,0.13,epochs)
    for i in range(9):
        t1=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        t0=(i+1)*365.25+t1
        t=np.append(t,t0)
    return t

def mainprocessing(m):
    tau=10*1.122**m
    at=np.array([])
    for k in range(N):
        name=str(round(ln[k]))
        t0,e,mean=SP(name)
        np.random.seed(m*N+k)
        t1=obs(cad)+25.75*365.25
        t2=np.concatenate((t0,t1))
        t=np.array([])
        for i in range(len(t2)):
            if t2[i]<365.25*(25+period):
                t=np.append(t,t2[i])
        e=np.concatenate((e,np.full(len(t)-len(t0),si)))
        y=dl.DRW_process(t,tau,0.2,mean)
        s=np.random.normal(y,e)
        try:
            re=dl.DRW_fit(t,s,e)
        except:
            continue
        at=np.append(at,re[0]/tau)
    print('[',m,']',tau)
    return np.concatenate(([tau],at))

#BEGIN
si=0.014
cad=6
period=1
# /home/charon/Desktop/MyLibrary/大创运行代码/Simulation/
root='/home/xfhu/data3/MyLibrary/大创运行代码/Simulation/'

ln=np.loadtxt(root+'Content')
N=len(ln)
print('N=',N,'sigma=',si,'Cadence=',cad,'T=',period)
if __name__ == '__main__':
    pool=Pool(41)
    l=list(pool.map(mainprocessing,range(41)))
    pool.close()
    pool.join()    
np.savetxt('c6t1e1.txt',np.array(l),fmt='%.4f')
print('time=',time.perf_counter()-start)
#END
