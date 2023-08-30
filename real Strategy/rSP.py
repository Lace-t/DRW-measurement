import numpy as np
from multiprocessing import Pool
import math
import time
import DRW_library as dl
start=time.perf_counter()
    
def MJD_trans(JD):
	JD = JD +2400001  # 转换为儒略历
	Z = math.floor(JD)
	F = JD - Z  # 日的小数部分
	if Z < 2299161:  # 儒略历
		A = Z
	else:  # 格里历
		a = math.floor((Z - 2305507.25) / 36524.25)
		A = Z + 10 + a - math.floor(a/4)
	k = 0
	while True:
		B = A + 1524  # 以BC4717年3月1日0时为历元
		C = math.floor((B-122.1)/365.25)  # 积年
		D = math.floor(365.25 * C)  # 积年的日数
		E = math.floor((B-D)/30.6)  # B-D为年内积日，E即月数
		day = B - D - math.floor(30.6*E) + F
		if day >= 1: break  # 否则即在上一月，可前置一日重新计算
		A -= 1
		k += 1
	month = E - 1 if E < 14 else E - 13
	year = C - 4716 if month > 2 else C - 4715
	day += k
	if int(day)==0:
		day+=1
	return (year+(month-1)/12+day/365.25)

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
        lt_sdss.append(MJD_trans(l[i][6]))
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
        lt_ps.append(MJD_trans(l[i][0]))
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

def mainprocessing(m):
    tau=10*1.122**m
    at=np.array([])
    for k in range(N):
        name=str(round(ln[k]))
        t,e,mean=SP(name)
        np.random.seed(seed[m*N+k])
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
root='/home/xfhu/data3/MyLibrary/大创运行代码/Simulation/'
seed=np.random.randint(0,2**31-1,9255*42)
ln=np.loadtxt(root+'Content')
N=9254
if __name__ == '__main__':
    pool=Pool(41)
    l=list(pool.map(mainprocessing,range(41)))
    pool.close()
    pool.join()    
np.savetxt('t0.txt',np.array(l),fmt='%.4f')
print('time=',time.perf_counter()-start)
#END
