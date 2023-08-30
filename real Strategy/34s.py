import numpy as np
import matplotlib.pyplot as plt
import math

def DRW_process(t,tau,m,SF):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)
    
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
    f=open('/home/charon/桌面/AGN光学光变的模拟和测量/Simulation/QSO_S82/'+name)
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
    f=open('/home/charon/桌面/AGN光学光变的模拟和测量/Simulation/PS1/'+name)
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
    return (t_sdss-1998)*365.25,sigma_sdss,(t_ps-1998)*365.25,sigma_ps,mean

def obs(cad):
    if cad in [5,10,30]:
        t=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
        for i in range(9):
            t1=np.linspace(1,91,cad*3+1)[:-1]+np.random.uniform(-0.13,0.13,cad*3)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
        return t
    elif cad==60:
        t=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
        for i in range(9):
            t1=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
            t0=(i+1)*365.25+t1
            t=np.append(t,t0)
            
        for i in range(10):
            t1=np.linspace(1,91,91)[:-1]+np.random.uniform(-0.13,0.13,90)
            t0=i*365.25+t1
            t=np.append(t,t0)    
        return np.sort(t)

tau=575
name=input('输入ID：')
t_sdss,e_sdss,t_ps,e_ps,mean=SP(name)
t_wfst=obs(5)+25.75*365

#dt=np.array([100,100,100,100,100,200,200,200,300,300,400,400,400,700])
#t_wfst=np.cumsum(np.concatenate((np.array([9131]),dt)))
t0=np.arange(0,365.25*35,1)
np.random.seed(5)
y=DRW_process(t0,tau,mean,0.2)
print('SDSS+PS1的采样数为',len(t_sdss)+len(t_ps))

s_sdss=np.array([])
for i in range(len(t_sdss)):
    j=round(t_sdss[i])
    s_sdss=np.append(s_sdss,np.random.normal(y[j],e_sdss[i],1)[0])
s_ps=np.array([])
for i in range(len(t_ps)):
    j=round(t_ps[i])
    s_ps=np.append(s_ps,np.random.normal(y[j],e_ps[i],1)[0])
s_wfst=np.array([])
e_wfst=np.array([])
for i in range(len(t_wfst)):
    j=round(t_wfst[i])
    
    e_r=0.03
    e_wfst=np.append(e_wfst,e_r)
    s_wfst=np.append(s_wfst,np.random.normal(y[j],e_wfst[-1],1)[0])
    #s_wfst=np.append(s_wfst,np.random.normal(y[j],0.03,1)[0])
    
plt.figure(figsize=(25,5),dpi=150)
plt.plot(t0/365.25+1998,y,'0.8',lw=1)
plt.errorbar(t_sdss/365.25+1998,s_sdss,yerr=e_sdss,fmt='o',ecolor='r',color='r',
             elinewidth=2,capsize=4,label='SDSS')
plt.errorbar(t_ps/365.25+1998,s_ps,yerr=e_ps,fmt='o',ecolor='g',color='g',
             elinewidth=2,capsize=4,label='PS1')
plt.errorbar(t_wfst/365.25+1998,s_wfst,yerr=e_wfst,fmt='o',ecolor='b',color='b',
             elinewidth=2,capsize=4,label='WFST')
plt.xlabel('t/yr',fontsize=15)
plt.ylabel('m/mag',fontsize=15)
plt.title(r'SDSS+PS1+WFST($\tau=575$ d,$\sigma=0.14$ mag,15 epochs/yr,$\epsilon=0.03$ mag)',fontsize=15)
plt.xlim(2000,2034)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15) 
plt.legend(fontsize=15)
print(t_wfst/365.25+1998)
plt.show()
