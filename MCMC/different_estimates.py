import numpy as np
import emcee
import DRW_library as dl
import celerite
from celerite import terms
from scipy.optimize import minimize
from multiprocessing import Pool
from time import perf_counter
start=perf_counter()

def DRW_fit(t,s,err,mean):
     # Define a cost function
    def mle(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def max_ap(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)+0.5*params[0]-params[1]*0.5    

    def log_probability(params):
        gp.set_parameter_vector(params)
        lp=gp.log_prior()
        loga=gp.get_parameter_dict().get('kernel:log_a')
        logc=gp.get_parameter_dict().get('kernel:log_c')
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(s)+lp-0.5*loga+logc*0.5
    
    # Set up the GP model
    bounds=dict(log_a=(2*np.log(0.02),2*np.log(0.7)),log_c=(-np.log(5000),0))
    kernel=terms.RealTerm(log_a=np.log(0.1414),log_c=-np.log(400),bounds=bounds)
    gp=celerite.GP(kernel,mean=mean,fit_mean=True)
    gp.compute(t,err)

    initial_params = gp.get_parameter_vector()
    soln = minimize(mle,initial_params,method="L-BFGS-B",args=(s,gp))
    gp.set_parameter_vector(soln.x)

    rt=np.exp(-gp.get_parameter_dict().get('kernel:log_c'))
    rs=np.exp(gp.get_parameter_dict().get('kernel:log_a')/2)
    #MCMC
    initial=np.array(soln.x)
    ndim, nwalkers=len(initial),16
    sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    #print("Running burn-in...")
    p0=initial+1e-4*np.random.randn(nwalkers,ndim)
    p0,lp,_=sampler.run_mcmc(p0,125)
    #print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0,500)
    
    lt_chain=np.sort(-sampler.flatchain[:,1])
    ls_chain=np.sort(sampler.flatchain[:,0]/2)
    t_e=np.mean(np.exp(lt_chain));s_e=np.mean(np.exp(ls_chain))
    t_m=np.exp(np.median(lt_chain));s_m=np.exp(np.median(ls_chain))

    bounds=dict(log_a=(2*np.log(0.02),2*np.log(0.7)),log_c=(-np.log(5000),0))
    kernel=terms.RealTerm(log_a=np.log(0.1414),log_c=-np.log(400),bounds=bounds)
    gp=celerite.GP(kernel,mean=mean,fit_mean=True)
    gp.compute(t,err)
 
    initial_params = gp.get_parameter_vector()
    soln=minimize(max_ap,initial_params,method="L-BFGS-B",args=(s,gp))
    gp.set_parameter_vector(soln.x)
 
    t_map=np.exp(-gp.get_parameter_dict().get('kernel:log_c'))
    s_map=np.exp(gp.get_parameter_dict().get('kernel:log_a')/2)
    return np.array([rt,t_map,t_e,t_m,rs,s_map,s_e,s_m])

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
            #s=np.sqrt(0.004**2+np.exp(1.63*(y[i]-22.55)))
            s=np.sqrt(0.013**2+np.exp(2*(y[i]-23.36)))
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
epochs=60
N=2
seed=np.random.randint(1,2**31-1,200*N)
print('epochs=',epochs,'N=',N)
if __name__ == '__main__':
    pool=Pool(61)
    l=np.array(pool.map(mainprocessing,range(61)))
    pool.close()
    pool.join()
print(l.shape)
np.save('de60j2.npy',l)
print('time=',perf_counter()-start)
#END    
