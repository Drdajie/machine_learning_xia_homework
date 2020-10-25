import numpy as np
import math
import sys
sys.path.append('../')
import Tools.Normalization as nm
import matplotlib.pyplot as plt
import random

class Softmax_Regression:
    """类中包含多种方法实现 softmax regression
    """
    def __init__(self,trainDataFile,classNum):
        """初始化
        1_读取 training data，并处理数据。
        2_初始化参数
        :param trainDataFile: 训练数据文件名
        :param testDataFile:  测试数据文件名
        :param clfNum: classfication num
        """
        #1_读取 training data，并处理
        #1.1_读取
        self.dataNum,self.dataX,self.dataY = self.load_dataFile(trainDataFile)
        self.classNum = classNum
        #1.2_处理
        self.xMin = self.dataX.min(axis=0)
        self.xMax = self.dataX.max(axis=0)
        self.dataX = nm.min_max_normalization(self.dataX,self.xMin,self.xMax)
        addDataX = np.ones((self.dataNum,1))

        self.dataX = np.hstack((addDataX,self.dataX)).T
        self.dataY = np.atleast_2d(self.dataY).T
        #2_初始化参数
        self.thetas = np.zeros((self.dataX.shape[0],classNum))

    def load_dataFile(self,dataFileName = ''):
        """读取文件
        :param dataFileName: 数据文件名
        :return: 存储数据的一个 ndarray 类型的变量
        """
        dataX = np.loadtxt(dataFileName + 'x.txt')
        dataY = np.loadtxt(dataFileName + 'y.txt')
        num = dataX.shape[0]
        return num,dataX,dataY

    def hypothesis(self,data,theClass):
        """预测结果的概率
        :param data: 待预测数据（ndarray)
        :return: 预测结果，如果 data 包含多个数据则返回 ndarray（一维）；
                 如果 data 只包含一个数据则返回一个数 float。
        """
        #分情况讨论，up、under分别代表分子分母
        if data.ndim == 1:
            up = 0.
            under = 0.
            #计算分母
            for j in range(self.classNum):
                under += math.exp(self.thetas[:, j] @ data)
            #计算分子
            up = math.exp(self.thetas[:,theClass] @ data)
        else:
            colNum = data.shape[1]
            up = np.zeros((colNum,))  # 分子
            under = np.zeros((colNum,))  # 分母
            # 计算分母
            for i in range(colNum):
                for j in range(self.classNum):
                    under[i] += math.exp(self.thetas[:, j] @ data[:, i])
            # 计算分子
            for i in range(colNum):
                up[i] = math.exp(self.thetas[:, int(theClass)] @ data[:, i])
        #计算结果
        ans = up/under
        return ans

    def log_likelihood(self):
        """对数自然函数，来衡量当前参数的好坏
        :return:一个浮点数，代表参数（model）的好坏。
        """
        l = 0.
        for i in range(self.dataNum):
            tempClass = self.dataY[i][0]
            tempData = np.atleast_2d(self.dataX[:,i]).T
            l += math.log(self.hypothesis(tempData,tempClass))
        return l

    def get_classResult(self,data):
        """计算 data 对应的预测结果
        思路:
            首先计算出每个数据对应每种类别的概率，然后再选出概率最大的那种作为结果。
        :param data: 带预测数据
        :return: 一个整型变量 或 一个元素为整型的一维 ndarray
        """
        #1_计算每个数据每种类别的概率
        if data.ndim == 1:
            ans = 0
            tempMax = 0
            for i in range(self.classNum):
                tempResult = self.hypothesis(data,i)
                if tempResult > tempMax:
                    tempMax = tempResult
                    ans = i
        else:
            colNum = data.shape[1]
            ans = np.zeros((colNum,))
            tempMax = np.zeros((colNum,))
            for i in range(colNum):
                for j in range(self.classNum):
                    tempResult = self.hypothesis(data[:,i],j)
                    if tempResult > tempMax[i]:
                        tempMax[i] = tempResult
                        ans[i] = j
        return ans

    def get_accuracy(self,hAns):
        """得出最后分类的准确度
        思路：
            用预测正确的个数除以全部的个数。
        :param hAns: 分类结果
        :return: 准确率
        """
        up = 0; under = self.dataNum     #分别为分子分母
        for i in range(self.dataNum):
            if hAns[i] == self.dataY[i][0]:
                up += 1
        return up/under

    def plot_trainResult(self,i,step,loss,accuracy):
        """画出当前结果
        思路：
            将整张图分割成许多点，对每个点对应的坐标进行预测，并用散点图显示。
        :param i: 当前迭代回数
        :param step: 图像中显示的迭代次数
        :param loss: 图像中显示的 loss 值
        :param accuracy: 图像中显示的准确率
        :return: tempAccuracy
        """
        bg = ['b', 'y', 'r']; mk = ['+', '^', 'o']; cs = ['k', 'g', 'o']
                                    # 分别代表背景颜色、散点标记、散点颜色的取值可能
        # 1_绘制 loss 图
        fig = plt.figure()
        plt.title('softmax train')
        plt.subplot(131)
        plt.title('loss')
        plt.xlabel("time")
        plt.ylabel("loss")
        step.append(i)  # 每幅图的x轴的刻度
        loss.append(self.log_likelihood())  # 每幅图的y轴的刻度
        plt.plot(step, loss)
        # 2_绘制准确率图
        plt.subplot(132)
        plt.title('accuracy')
        plt.xlabel('time')
        plt.ylabel('accuracy')
        hAns = self.get_classResult(self.dataX)
        tempAccuracy = self.get_accuracy(hAns)
        accuracy.append(tempAccuracy)
        plt.plot(step, accuracy)
        # 3_绘制分类图
        plt.subplot(133)
        # 绘制背景
        # 初始化定义参数，易于之后修改
        xL = 0;
        xR = 1  # 分别代表散点图 x 轴的左右范围
        yL = 0;
        yH = 1  # 分别代表散点图 y 轴的下上范围
        tempRange = 100  # 代表要将 x、y 轴分为多少段
        meshX, meshY = np.meshgrid(np.linspace(xL, xR, tempRange),
                                   np.linspace(yL, yH, tempRange))
        # 注意 meshX、meshY 都是二维的
        tempAdd = np.ones((meshX.size,))
        meshData = np.vstack((tempAdd, meshX.flatten()))
        meshData = np.vstack((meshData, meshY.flatten()))
        meshPrdic = self.get_classResult(meshData)
        plt.contourf(meshX, meshY, meshPrdic.reshape(meshX.shape))
        # 绘制散点
        plt.xlabel('x1')
        plt.ylabel('x2')
        for j in range(self.dataNum):
            plt.scatter(self.dataX[1, j], self.dataX[2, j],
                        marker=mk[int(self.dataY[j][0])],
                        c=cs[int(self.dataY[j][0])])
        return tempAccuracy

    def GD_train(self):
        """用 gradient descent 方法训练模型；显示训练过程中 loss、准确率、分类情况的变化。
        思路：
            因为 softmax 方法是针对每一种类型都有一组参数与之对应，要注意分开考虑。
        步骤：
            1_计算每种类型对应的那组参数的 error * feature
            2_更新参数
            3_画图像
        :return: 无
        """
        #初始化参数
        a = 0.01                                 #a为 learning rate
        step = []; accuracy = []; loss = []      #分别为迭代次数、准确率、loss大小
        def error_feature(tempTheta):
            """计算 error * feature
            :param tempTheta: (int) 表示当前更改的参数
            :return: (float) error * feature 结果
            """
            ans = np.zeros((self.dataX.shape[0],))
            for i in range(self.dataNum):
                x = self.dataX[:,i]
                if self.dataY[i][0] == tempTheta:
                    ans += (1 - self.hypothesis(self.dataX[:,i],tempTheta)) * x
                else:
                    ans += -self.hypothesis(self.dataX[:,i],tempTheta) * x
            return ans
        #画图
        i = 0
        plt.ion()
        while 1:
            # 1_计算每种类型对应的那组参数的 error * feature 并更新参数
            for j in range(self.classNum):
                self.thetas[:, j] += a * error_feature(j)
            tempAccuracy = self.plot_trainResult(i,step,loss,accuracy)
            # “善后处理”
            i = i + 1
            plt.pause(0.001)
            if tempAccuracy > 0.8:
                break
        plt.ioff()
        plt.show()

    def SGD_train(self):
        """用 stochastic gradient descent 方法训练模型；显示训练过程中 loss、准确率、分类情况的变化。
        思路：
            因为 softmax 方法是针对每一种类型都有一组参数与之对应，要注意分开考虑。
        步骤：
            1_计算每种类型对应的那组参数的 error * feature
            2_更新参数
            3_画图像
        :return: 无
        """
        #初始化参数
        a = 0.03                                 #a为 learning rate
        step = []; accuracy = []; loss = []      #分别为迭代次数、准确率、loss大小
        def error_feature(tempDataIndex,tempTheta):
            """计算 error * feature
            :param tempDataIndex: (int) 表示当前更新所用的数据
            :param tempThetaIndex: (int) 表示当前更改的参数
            :return: (float) error * feature 结果
            """
            ans = 0.
            x = self.dataX[:,tempDataIndex]
            if self.dataY[tempDataIndex][0] == tempTheta:
                ans = (1 - self.hypothesis(self.dataX[:,tempDataIndex],tempTheta)) * x
            else:
                ans = -self.hypothesis(self.dataX[:,tempDataIndex],tempTheta) * x
            return ans
        #画图
        i = 0
        plt.ion()
        while 1:
            # 1_计算每种类型对应的那组参数的 error * feature 并更新参数
            k = random.randint(0,self.dataNum-1)          #k 代表当前更新使用的数据
            for j in range(self.classNum):
                self.thetas[:, j] += a * error_feature(k,j)
            tempAccuracy = self.plot_trainResult(i,step,loss,accuracy)
            # “善后处理”
            i = i + 1
            plt.pause(0.001)
            if tempAccuracy > 0.8:
                break
        plt.ioff()
        plt.show()