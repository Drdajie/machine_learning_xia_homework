import numpy as np

def min_max_normalization(dataX):
    #alter dataX to ndarray
    tempType = type(dataX)
    if tempType == 'numpy.matrix':
        x = np.asarray(dataX)
    elif tempType == 'list':
        x = np.array(dataX)
    else:
        x = dataX

    #process
    myMin = np.min(x); myMax = np.max(x)
    x = (x - myMin)/(myMax - myMin)

    return x
