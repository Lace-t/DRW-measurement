import os
import numpy as np

N=1000
for i in np.arange(0,151,4):
    for j in range(N):
        try:
            os.remove('lc'+str(i)+'_'+str(j))
        except:
            continue
