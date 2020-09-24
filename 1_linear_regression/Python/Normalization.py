import numpy as np

def min_max_normalization(dataX):
    x = dataX

    #process
    myMin = np.min(x); myMax = np.max(x)
    x = (x - myMin)/(myMax - myMin)
    #print(dataX)
    return x