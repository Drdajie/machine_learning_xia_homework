import numpy as np
import math

def sigmoid(data):
    if(type(data) == 'numpy.ndarray'):
        temp = np.exp(data)
        if(data.ndim == 2):
            return temp[0]
        else:
            return temp
    else:
        return math.exp(data)