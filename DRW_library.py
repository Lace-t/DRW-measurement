import numpy as np
import emcee
import celerite
from celerite import terms
from numpy.linalg import svd
from scipy.optimize import minimize

def DRW_process(t,tau,SF,m):
    '''Return a DRW light curve

    parameters
    ----------
    t  : array_like
         epcohs
    tau: float
         input timescale
    SF : float
         input StructureFunction_infty(=sqrt(2)*amplitude)
    m  : float
         mean of the ligh curve

    Returns
    -------
    DRW_process:ndarray
                a simulated light curve
    '''   
    r=np.diff(t)/tau
    ls=[np.random.normal(m,SF/1.414,1)[0]]
    for i in range(len(t)-1):
        if r[i]<0:print('Error:时间序列未排序');return 
        stdev=(1-np.exp(-2*r[i]))**0.5*SF/1.414
        loc=ls[i]*np.exp(-r[i])+m*(1-np.exp(-r[i]))
        ls.append(np.random.normal(loc,stdev,1)[0])
    return np.array(ls)

def log_like(params,y,gp,method,prior):
    gp.set_parameter_vector(params)
    if method=='MAP':
        if prior=='K17':
            return -gp.log_likelihood(y)+0.5*params[0]-params[1]*0.5
        else:
            return -gp.log_likelihood(y)+0.5*params[0]-params[1]
    else: return -gp.log_likelihood(y)

def log_probability(params,y,gp,prior):
        gp.set_parameter_vector(params)
        lp=gp.log_prior()
        log_a=gp.get_parameter_dict().get('kernel:log_a')
        log_c=gp.get_parameter_dict().get('kernel:log_c')
        if not np.isfinite(lp):
            return -np.inf
        if prior=='K17':
            return gp.log_likelihood(y)+lp-0.5*log_a+log_c*0.5
        else:
            return gp.log_likelihood(y)+lp-0.5*log_a+log_c

def DRW_fit(t,s,err,method='PM',prior='K17',tl=[1,5000],sl=[0.02,0.7],uncertainty=False):
    '''Return the fitting timescale and amplitude of the DRW model

    Parameters
    ----------
    t                    : array-like
                           observation time
    s                    : array-like
                           observation magnitude
    err                  : array-like
                           manitude uncertainty
    method(optional)     : str
                           "MLE": maximum likelihood esimate
                           "MAP": maximum a posterior
                           "PE" : posterior expectancy
                           "PM" : posterior median
                           "MAX": max{PM,MLE}(recommended when epochs are not enough)
                           (default: "PM")
    prior                : str
                           K17: prior=1/(sigma*sqrt(tau)) [Kozlowski 2017 A&A]
                           S21: prior=1/(sigma*tau) [Suberlak et al. 2021 ApJ]
                           (default: "K17")
    tl(optional)         : list with 2 elements
                           boundary of tau_min and tau_max
                           only useful for "MLE","MAP"
                           (default: [1,5000])
    sl(optional)         : list with 2 elements
                           boundary of sigma_min and sigma_max
                           only useful for "MLE","MAP"
                           (default: [0.02,0.7])
    uncertainty(optional): "True" or "False"
                           "True": return the fitting result and uncertainty. 
                                   For "PE", it also returns standard deviation of posterior.
                                   For "PM", it also returns 16% and 84% of posterior.
                           (default: "False")

    Returns
    -------
    DRW_fit: ndarray
    
    uncertainty=='False': np.array([tau,sigma])
    uncertainty=='True' and method=='PE': np.array([tau,tau_err,sigma,sigma_err])
    uncertainty=='True' and method=='PM': np.array([tau,tau_16%,tau_84%,sigma,sigma_16%,sigma_84%])
     ''' 
    # Set up the GP model
    bounds=dict(log_a=(2*np.log(sl[0]),2*np.log(sl[1])),
                log_c=(-np.log(tl[1]),-np.log(tl[0])))
    kernel=terms.RealTerm(log_a=np.log(0.1414),log_c=-np.log(400),bounds=bounds)
    gp=celerite.GP(kernel,mean=np.mean(s),fit_mean=True)
    gp.compute(t,err)
    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    soln=minimize(log_like, initial_params,method="L-BFGS-B",args=(s,gp,method,prior))
    
    gp.set_parameter_vector(soln.x)
    rt=np.exp(-gp.get_parameter_dict().get('kernel:log_c'))#MLE or MAP
    rs=np.exp(gp.get_parameter_dict().get('kernel:log_a')/2)
    if method=='MLE' or method=='MAP':
        return np.array([rt,rs])   
    #MCMC
    initial=np.array(soln.x)
    ndim, nwalkers=len(initial),16
    sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(s,gp,prior))
    #Running burn-in
    p0=initial+1e-4*np.random.randn(nwalkers,ndim)
    p0,lp,_=sampler.run_mcmc(p0,125)    
    #Running production
    sampler.reset()
    sampler.run_mcmc(p0,500)
    #Markov chains
    t_chain=np.exp(np.sort(-sampler.flatchain[:,1]))
    s_chain=np.exp(np.sort(sampler.flatchain[:,0]/2))
    if method=='PE':
        rt=np.mean(t_chain)
        rs=np.mean(s_chain)
        if uncertainty:
            return np.array([rt,np.std(t_chain),rs,np.std(s_chain)])
        else:
            return np.array([rt,rs])
    elif method=='PM':
        rt=np.median(t_chain)
        rs=np.median(s_chain)
        if uncertainty:
            return np.array([rt,np.percentile(t_chain,16),np.percentile(t_chain,84),
                             rs,np.percentile(s_chain,16),np.percentile(s_chain,84)])
        else:
            return np.array([rt,rs])
    elif method=='MAX':
        t_m=np.median(t_chain)
        s_m=np.median(s_chain)
        return np.array([max(t_m,rt),max(s_m,rs)])

def GP_process(t,tau,sigma,m,beta=1):
    #covariance matrix
    cov=np.zeros((len(t),len(t)))
    for i in range(len(t)):
        for j in range(len(t)):
            cov[i,j]=sigma**2*np.exp(-(abs((t[i]-t[j])/tau))**beta)
    #SVD decomposition
    u,sv,v=svd(cov)
    s=np.diag(sv)
    #generate value
    n=np.random.normal(0,1,len(t))
    y=m+np.dot(np.dot(u,np.sqrt(s)),n)
    return y

def MJDtoCE(JD):
    '''Convert MJD to Common Era[unit:year]

    parameters
    ----------
    JD: float
        MJD time

    Returns
    -------
    MJDtoJD: float
             Common Era time
    '''
    JD = JD +2400001  # 转换为儒略历
    Z = np.floor(JD)
    F = JD - Z  # 日的小数部分
    if Z < 2299161:  # 儒略历
            A = Z
    else:  # 格里历
            a =np.floor((Z - 2305507.25) / 36524.25)
            A = Z + 10 + a - np.floor(a/4)
    k = 0
    while True:
            B = A + 1524  # 以BC4717年3月1日0时为历元
            C = np.floor((B-122.1)/365.25)  # 积年
            D = np.floor(365.25 * C)  # 积年的日数
            E = np.floor((B-D)/30.6)  # B-D为年内积日，E即月数
            day = B - D - np.floor(30.6*E) + F
            if day >= 1: break  # 否则即在上一月，可前置一日重新计算
            A -= 1
            k += 1
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    day += k
    if int(day)==0:
            day+=1
    return (year+(month-1)/12+day/365.25)

def chi(t,s,err,fit):
    '''Calculate chi-square
    Parameters
    ----------
    t:   array-like
         epochs
    s:   array-like
         observation magnitude
    err: array-like
         magnitude uncertainty
    fit: array-like
         tau_hat and sigma_hat
    Returns
    -------
    chi: float
         the value of chi-square
    '''
    kernel=terms.RealTerm(log_a=2*np.log(fit[1]),log_c=-np.log(fit[0]))
    gp=celerite.GP(kernel,mean=np.mean(s),fit_mean=True)
    gp.compute(t,err)

    mu,_=gp.predict(s,t,return_var=True)
    return np.mean(((s-mu)/err)**2)

def pred(t,s,err,fit):
    '''Return the prediction and its uncertainty
    Parameters
    ----------
    t:   array-like
         epochs
    s:   array-like
         observation magnitude
    err: array-like
         magnitude uncertainty
    fit: array-like
         tau_hat and sigma_hat
    Returns
    -------
    pred: ndarray

    mu : the prediction value
    std: the prediction uncertainty 
    '''
    kernel=terms.RealTerm(log_a=2*np.log(fit[1]),log_c=-np.log(fit[0]))
    gp=celerite.GP(kernel,mean=np.mean(s),fit_mean=True)
    gp.compute(t,err)

    mu,var=gp.predict(s,t,return_var=True)
    return mu,np.sqrt(var)
'''    
#调试代码
t=np.sort(np.random.uniform(0,2922,445))
y=DRW_process(t,29,0.2,18)
e=np.full_like(y,0.03)
s=np.random.normal(y,e)
re=DRW_fit(t,s,e)
print(re)
#print(chi(t,s,e,re))
'''
