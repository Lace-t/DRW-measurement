import numpy as np

#x=np.loadtxt('fort.60',dtype=str)

tau=np.array([])
for i in range(1,7):
    x=np.loadtxt('fort.445.'+str(i),dtype=str)
    tau=np.append(tau,10**x[:,8].astype('float'))
    
print(tau.shape)
#tau=10**x[:,8].astype('float')
#sigma=np.sqrt(tau/2)*10**x[:,5].astype('float')

tau.resize(61,10000)
np.savetxt('k445.txt',tau,fmt='%.4f')
