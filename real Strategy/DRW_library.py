import numpy as np
import emcee
import celerite
from celerite import terms
from scipy.optimize import minimize

def DRW_process(t,tau,SF,m):
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def log_like(params,y,gp,method):
    gp.set_parameter_vector(params)
    if method=='MAP':
        return -gp.log_likelihood(y)+0.5*params[0]-params[1]
    else: return -gp.log_likelihood(y)

def log_probability(params,y,gp):
        gp.set_parameter_vector(params)
        lp=gp.log_prior()
        log_a=gp.get_parameter_dict().get('kernel:log_a')
        log_c=gp.get_parameter_dict().get('kernel:log_c')
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(y)+lp-0.5*log_a+log_c

def DRW_fit(t,s,err,mean,method='MAX',tl=[1,5000],sl=[0.02,0.7]):
    
    # Set up the GP model
    bounds=dict(log_a=(2*np.log(sl[0]),2*np.log(sl[1])),
                log_c=(-np.log(tl[1]),-np.log(tl[0])))
    kernel=terms.RealTerm(log_a=np.log(0.1414),log_c=-np.log(400),bounds=bounds)
    gp=celerite.GP(kernel,mean=mean,fit_mean=True)
    gp.compute(t,err)
    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    soln=minimize(log_like, initial_params,method="L-BFGS-B",args=(s,gp,method))
    
    gp.set_parameter_vector(soln.x)
    rt=np.exp(-gp.get_parameter_dict().get('kernel:log_c'))#MLE or MAP
    rs=np.exp(gp.get_parameter_dict().get('kernel:log_a')/2)
    if method=='MLE' or method=='MAP':
        return np.array([rt,rs])   
    #MCMC
    initial=np.array(soln.x)
    ndim, nwalkers=len(initial),16
    sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(s,gp))
    #Running burn-in
    p0=initial+1e-4*np.random.randn(nwalkers,ndim)
    p0,lp,_=sampler.run_mcmc(p0,125)
    #Running production
    sampler.reset()
    sampler.run_mcmc(p0,500)
    #Markov chains
    lt_chain=np.sort(-sampler.flatchain[:,1])
    ls_chain=np.sort(sampler.flatchain[:,0]/2)
    if method=='PE':
        rt=np.exp(np.mean(lt_chain))
        rs=np.exp(np.mean(ls_chain))
        return np.array([rt,rs])
    elif method=='PM':
        median=len(lt_chain)//2-1
        rt=np.exp(lt_chain[median])
        rs=np.exp(ls_chain[median])
        return np.array([rt,rs])
    elif method=='MAX':
        t_e=np.exp(np.mean(lt_chain))
        s_e=np.exp(np.mean(ls_chain))
        return np.array([max(t_e,rt),max(s_e,rs)])
