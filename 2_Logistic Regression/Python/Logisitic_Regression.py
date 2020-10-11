import numpy as np
import sys
sys.path.append("../../")
import Tools.Common_Function as commonTools
import matplotlib.pyplot as plt

class Logistic_Regression:
    """用 logistic regression 来预测分类
    """
    def __init__(self,fileName = "../../Data/ex4Data"):
        """导入数据并做处理；创建参数向量并设初值；创建相关参数
        :param fileName: traning data 所在目录
        """
        #导入数据并做处理
        #导入数据: 样本个数，input 数据,label 数据
        self.dataNum,self.dataX,self.dataY = self.load_data(fileName)
        #处理
        #1.1_为每个 input 增加一个值为 1 的元素，方便之后与 参数做内积。并得到参数维度
        self.dataX = commonTools.join_biasTerm(self.dataNum,self.dataX)
        self.vDim = self.dataX.shape[1]
        #1.2_转换 xData，使其中的每个 input 向量均为列向量
        self.dataX = self.dataX.T
        #2.3_转换 yData，使其变成列向量。
        #因为 yData 是一维的，无法转置，所以先要更改其维度再进行转置操作。
        self.dataY = np.atleast_2d(self.dataY).T
        #初始化参数
        self.thetas = np.zeros((self.vDim, 1))


    def load_data(self,fileName):
        """用于将 training data 的 input 和 label 分别读入
        思路：
            先将文件中的数据存储于列表当中，再将列表转化为 numpy.ndarray 类型。
        :param fileName: traning data 所在目录
        :return: training data 中的样本数量，ndarray 类型的 input、label 数据
        """
        #初始化工具列表和最终要返回的样本个数
        tempXList = []
        tempYList = []
        dataNum = 0
        #1.1_读取 input 数据，存储在 tempXList，并把它转化为 ndarray类型的 xData 变量。（顺便统计dataNum）
        with open(fileName + "/ex4x.dat","r") as xFile:
            for line in xFile.readlines():
                dataNum = dataNum + 1
                tempTool = line.strip().split()
                for i in range(len(tempTool)):
                    tempTool[i] = float(tempTool[i])
                tempXList.append(tempTool)
        #1.2_转换 tempXList
        xData = np.array(tempXList)
        #2.1_读取 label 数据，存储在 tempYList，并把它转化为 ndarray类型的 yData 变量。
        with open(fileName + "/ex4y.dat","r") as yFile:
            for line in yFile.readlines():
                tempYList.append(float(line.strip()))
        #2.2_转换 tempYList。
        yData = np.array(tempYList)
        return dataNum,xData,yData

    def hypothesis(self,X):
        return commonTools.sigmoid(X.T @ self.thetas)

    def likelihood_estimation(self):
        pass

    def GD_getAns(self):
        """用Gradient descent完成任务
        """
        #初始化要用的东西（包括函数和参数等等）
        def error_feature():
            """error * feature -> 代表着‘求导’之后的东西"""
            return self.dataX @ (self.dataY - self.hypothesis())
        a = 0.01                                            #learning rate
        iNum = 100                                          #iteration num
        loss = [];step = []                                 #loss 图的坐标轴
        #process & plot
        plt.ion()
        for i in range(iNum):
            #GD process
            self.thetas = self.thetas + a * error_feature()
            plt.cla()
            #绘制 loss 图
            plt.subplot(121)

