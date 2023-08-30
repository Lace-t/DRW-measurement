import numpy as np
import emcee
import celerite
from celerite import terms
from scipy.optimize import minimize
from multiprocessing import Pool
from time import perf_counter
start=perf_counter()

def DRW_process(t,tau,m,SF):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def DRW_fit(t,s,err,mean):
     # Define a cost function
    def neg_log_like(params,s,gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(s)

    def grad_neg_log_like(params,s,gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(s)[1]

    def log_probability(params):
        gp.set_parameter_vector(params)
        lp=gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        #print(gp.log_likelihood(s),gd.get('kernel:log_a'),gd.get('kernel:log_c'))
        return gp.log_likelihood(s)+lp-0.5*gd.get('kernel:log_a')+gd.get('kernel:log_c')

    # Set up the GP model
    bounds=dict(log_a=(-np.inf,0),log_c=(-14,+np.inf))
    kernel=terms.RealTerm(log_a=np.log(0.1414),log_c=-np.log(400),bounds=bounds)
    gp=celerite.GP(kernel,mean=mean,fit_mean=True)
    gp.compute(t,err)
    #print("Initial log-likelihood: {0}".format(gp.log_likelihood(s)))
    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    soln = minimize(neg_log_like,initial_params,jac=grad_neg_log_like,
                    method="L-BFGS-B",args=(s,gp))
    gp.set_parameter_vector(soln.x)
    #print("Final log-likelihood: {0}".format(-soln.fun))
    gd=gp.get_parameter_dict()
    rt=np.exp(-gd.get('kernel:log_c'))
    rs=np.exp(gd.get('kernel:log_a')/2)
    #MCMC
    t_e=0
    while t_e<0.1:
        initial=np.array(soln.x)
        ndim, nwalkers=len(initial),16
        sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability)
        #print("Running burn-in...")
        p0=initial+1e-8*np.random.randn(nwalkers,ndim)
        p0,lp,_=sampler.run_mcmc(p0,500)
        #print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0,2000)
    
        lt_chain=np.sort(-sampler.flatchain[:,1])
        ls_chain=np.sort(sampler.flatchain[:,0]/2)
        t_e=np.exp(np.mean(lt_chain));s_e=np.exp(np.mean(ls_chain))
    return t_e

def mainprocessing(m):
    np.random.seed(seed[m])
    t0=np.sort(np.random.randint(0,13*tau,epochs)+np.random.uniform(-0.13,0.13,epochs))
    dt=np.diff(t0)
    np.random.shuffle(dt)
    t1=np.cumsum(np.r_[t0[0],dt])
    t2=np.linspace(0.13,13*tau,epochs)+np.random.uniform(-0.13,0.13,epochs)
    lt0=[];lt1=[];lt2=[]
    for k in range(N):
        np.random.seed(seed[N*m+k+100])
        #Initial
        le=[];ls=[];
        rt=999;rs=999
        y=DRW_process(t0,tau,18,0.2)
        rt=DRW_fit(t0,y,np.zeros(epochs),np.mean(y))
        lt0.append(rt)
        #Changed
        le=[];ls=[];
        rt=999;rs=999
        y=DRW_process(t1,tau,18,0.2)
        rt=DRW_fit(t1,y,np.zeros(epochs),np.mean(y))
        lt1.append(rt)
        #Uniform
        le=[];ls=[];
        rt=999;rs=999
        y=DRW_process(t1,tau,18,0.2)
        rt=DRW_fit(t1,y,np.zeros(epochs),np.mean(y))
        lt2.append(rt)
    at0=np.array(lt0);
    at1=np.array(lt1);
    at2=np.array(lt2);
    i1=round(np.mean(at0),4);i2=round(np.std(at0),4);
    c1=round(np.mean(at1),4);c2=round(np.std(at1),4);
    u1=round(np.mean(at2),4);u2=round(np.std(at2),4);
    
    print('\n[',m,']')
    print('Initial: t=',i1,'std_t=',i2)
    print('Changed: t=',c1,'std_t=',c2)
    print('Uniform: t=',u1,'std_t=',u2)
    
    output=open('Initial.txt','a+')
    output.write(str(i1)+' '+str(i2))
    output.write('\n')
    output.close()
    output=open('Changed.txt','a+')
    output.write(str(c1)+' '+str(c2))
    output.write('\n')
    output.close()
    output=open('Uniform.txt','a+')
    output.write(str(u1)+' '+str(u2))
    output.write('\n')
    output.close()
    
#BEGIN
N=10000
epochs=75
tau=200
seed=np.random.randint(1,2**31-1,N*100+100)
print('epochs=',epochs,'N=',N,'tau=',tau)
if __name__ == '__main__':
    pool=Pool(60)
    pool.map(mainprocessing, range(60))
    pool.close()
    pool.join()    

print('time=',perf_counter()-start)
#END
