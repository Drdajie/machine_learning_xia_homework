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
    x = (x - myMin)/(myMax - myMin)
    return x

def reverse_mm_normalization(data,myMin,myMax):
    return data * (myMax-myMin) + myMin
