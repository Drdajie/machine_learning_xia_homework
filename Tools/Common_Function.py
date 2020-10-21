import numpy as np
import math

def sigmoid(data):
    """该函数为: 1/（1+e^-z)
    :param data: 一维或二维
    :return:
    """
    #input 参数可能为 ndarray 和 普通数字
    if(type(data) == np.ndarray):
        temp = np.exp(-data)
        return (1/(1 + temp))
    else:
        return 1/(1+math.exp(-data))

def join_biasTerm(dataNum,data):
    """
    机器学习中 linear model 中含有 bias 一项（即常数项），为了可以让 theta变量直接与input相乘，所以给 input 增加值为全 1 的一列
    思路：
    1.先根据 dataNum 创建一个全为 1 的向量 tempV
    2.将tempV 与 data 合并（列向量的形式）
    :param data: input 数据
    :param dataNum: 数据包含的样本个数
    :return: 增加了值全为 1 的 data
    """
    #1_创建全 1 向量
    tempV = np.ones((dataNum,1))
    #2_合并
    return np.hstack((tempV,data))



