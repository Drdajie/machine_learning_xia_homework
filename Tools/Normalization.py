import numpy as np

def min_max_normalization(data,myMin,myMax):
    #alter dataX to ndarray
    tempType = type(data)
    if tempType == 'numpy.matrix':
        x = np.asarray(data)
    elif tempType == 'list':
        x = np.array(data)
    else:
        x = data
    #process
    if x.ndim == 1:
        x = (x - myMin) / (myMax - myMin)
    else:
        n = x.shape[1]
        if n == 1:
            x = (x - myMin)/(myMax - myMin)
        if n == 2:
            for i in range(n):
                x[:,i] = (x[:,i] - myMin[i])/(myMax[i] - myMin[i])
    return x

def reverse_mm_normalization(data,myMin,myMax):
    return data * (myMax-myMin) + myMin
