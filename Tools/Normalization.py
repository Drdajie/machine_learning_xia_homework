import numpy as np

def min_max_normalization(data,myMin,myMax):
    """最大最小值归一化
    :param data: 待归一数据（ndarray）
    :param myMin: 数据最小值（如果 data 是多个列表，则输入包含多个元素的 ndarray 变量）
    :param myMax: 数据最大值（同上）
    :return: 归一化之后的数据（ndarray）
    """
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
