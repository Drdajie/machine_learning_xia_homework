import numpy as np
import sys
sys.path.append("../../")
import Tools.Common_Function as commonTools
import Tools.Normalization as nm
import matplotlib.pyplot as plt
import math
import random

class Logistic_Regression:
    """用 logistic regression 来预测分类
    """
    def __init__(self,fileX,fileY):
        """导入数据并做处理；创建参数向量并设初值；创建相关参数
        :param fileName: traning data 所在目录(不包含文件名)
        """
        #导入数据并做处理
        #导入数据: 样本个数，input 数据,label 数据
        self.dataNum,self.dataX,self.dataY = self.load_data(fileX,fileY)
        self.xMin = self.dataX.min(axis=0)
        self.xMax = self.dataX.max(axis=0)
        #处理
        #1.1_归一化
        self.dataX = nm.min_max_normalization(self.dataX, self.xMin, self.xMax)
        #1.2_为每个 input 增加一个值为 1 的元素，方便之后与 参数做内积。并得到参数维度
        self.dataX = commonTools.join_biasTerm(self.dataNum,self.dataX)
        #1.3_转换 xData，使其中的每个 input 向量均为列向量
        self.dataX = self.dataX.T
        #2.3_转换 yData，使其变成列向量。
        #因为 yData 是一维的，无法转置，所以先要更改其维度再进行转置操作。
        self.dataY = np.atleast_2d(self.dataY).T
        #初始化参数
        self.thetas = np.zeros((self.dataX.shape[0], 1))

    def load_data(self,fileX,fileY):
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
        with open(fileX) as xFile:
            for line in xFile.readlines():
                dataNum = dataNum + 1
                tempTool = line.strip().split()
                for i in range(len(tempTool)):
                    tempTool[i] = float(tempTool[i])
                tempXList.append(tempTool)
        #1.2_转换 tempXList
        xData = np.array(tempXList)
        #2.1_读取 label 数据，存储在 tempYList，并把它转化为 ndarray类型的 yData 变量。
        with open(fileY) as yFile:
            for line in yFile.readlines():
                tempData = float(line.strip())
                tempYList.append(int(tempData))
        #2.2_转换 tempYList。
        yData = np.array(tempYList)
        return dataNum,xData,yData

    def hypothesis(self,X):
        """通过Logistic Regression来预测 input X 对应的值。
        :param X: 一维或二维(向量或矩阵）
        :return: 若 X 为一维则返回一个float类型的值；若 X 为二维则返回一个列向量。
        """
        ans = commonTools.sigmoid(X.T @ self.thetas)
        if(ans.size == 1):
            return ans[0]
        else:
            return ans

    def get_loss(self,h):
        """
        :param h: hypothesis -> 预测结果
        :return: loss的值
        """
        loss = 0
        for i in range(self.dataNum):
            tempH, tempY = h[i], self.dataY[i]
            loss += math.pow(tempH, tempY) * math.pow(1 - tempH, 1 - tempY)
        return loss

    def likelihood_estimation(self):
        pass

    def get_classResult(self,h):
        """将预测结果转换为分类结果
        :param h: 预测结果（概率）
        :return: 分类结果
        """
        ans = []
        for i in range(h.shape[0]):
            if(h[i][0] >= 0.5):
                ans.append(1)
            else:
                ans.append(0)
        return ans

    def get_accuracy(self,h,dataY):
        """计算当前参数对应的预测准确率
        思路：
            用预测正确的个数除以全部的样本个数。
        :param h: 当前预测结果，类型为一个列向量，里面存储的是预测值（概率）
        :return: 准确率，一个数
        """
        #1_得到分类结果
        ans = self.get_classResult(h)
        #2_计算正确率
        cNum = 0                              #correct num
        for i in range(h.size):
            if ans[i] == dataY[i][0]:
                cNum += 1
        return cNum/h.size

    def plot_testResult(self,testXFile,testYFile = ''):
        """绘制分类图，由数据点和分类线构成。
        注意:
            1_在画图之前先对数据预处理
            2_数据点用红蓝两色来区分不同的分类
        :param testXFile: 测试数据 input 的文件名
        :param testYFile: 测试数据 output 的文件名
        :return: 无
        """
        if testYFile != '':
            #预处理
            dataNum,dataX,dataY = self.load_data(testXFile,testYFile)
            dataX = nm.min_max_normalization(dataX,self.xMin,self.xMax)
            add = np.ones((dataX.shape[0],1))
            dataX = np.hstack((add,dataX)).T
            dataY = np.atleast_2d(dataY).T
            # 1_绘制数据点
            plt.title('logistic test')
            plt.xlabel("x1")
            plt.ylabel("x2")
            for j in range(dataX.shape[1]):
                if dataY[j,0] == 1:
                    plt.plot(dataX[1,j], dataX[2,j], 'ro')
                else:
                    plt.plot(dataX[1,j], dataX[2,j], 'bo')
            # 2_绘制分类线
            # 思路：
            # 因为模型的预测结果是通过比较sigmoid函数的输出与0.5得到的。
            # 而当sigmoid函数output为0.5时，w0 + w1*x1 + w2*x2 = 0。
            # 又因为图像的x、y轴分别为参数 x1、x2。
            # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
            x = np.linspace(-0.5, 1.2, 50)
            y = -(self.thetas[0][0] + self.thetas[1][0] * x) / self.thetas[2][0]
            plt.plot(x, y, 'k')
            # 3_计算准确率
            hAns = self.hypothesis(dataX)
            plt.show()
            return self.get_accuracy(hAns,dataY)

    def GD_getAns(self):
        """用Gradient descent完成任务
        """
        #初始化要用的东西（包括函数和参数等等）
        def error_feature():
            """error * feature -> 代表着‘求导’之后的东西"""
            return self.dataX @ (self.dataY - self.hypothesis(self.dataX))
        a = 0.01                                            #learning rate
        loss = [];step = [];accuracy = []                   #loss图和准确率图的坐标轴
        #process & plot
        plt.ion()
        i = 0
        while 1 :
            #GD process
            self.thetas = self.thetas + a * error_feature()
            plt.cla()
            plt.suptitle('logistic train')
            #绘图,绘制 loss、准确率、拟合图
            #1_绘制 loss 图
            plt.subplot(131)
            plt.title('loss')
            plt.xlabel("time")
            plt.ylabel("loss")
            step.append(i)                                  #每幅图的x轴的刻度
            h_ans = self.hypothesis(self.dataX)             #h_ans 为traning data 的预测结果
            loss.append(self.get_loss(h_ans))               #每幅图的y轴的刻度
            plt.plot(step,loss)
            #2_绘制准确率图
            plt.subplot(132)
            plt.title('accuracy')
            plt.xlabel('time')
            plt.ylabel('accuracy')
            accuracy.append(self.get_accuracy(h_ans,self.dataY))
            plt.plot(step,accuracy)
            #3_绘制分类图，由数据点和分类线构成。
            #其中数据点用红蓝两色来区分不同的分类（该种类为trainning data 给定的）
            #3.1_绘制数据点
            plt.subplot(133)
            plt.title('classification')
            plt.xlabel("x1")
            plt.ylabel("x2")
            redPointX = []
            redPointY = []
            bluePointX = []
            bluePointY = []
            for j in range(self.dataNum):
                if self.dataY[j] == 1:
                    redPointX.append(self.dataX[1][j])
                    redPointY.append(self.dataX[2][j])
                else:
                    bluePointX.append(self.dataX[1][j])
                    bluePointY.append(self.dataX[2][j])
            plt.plot(redPointX, redPointY, 'ro')
            plt.plot(bluePointX,bluePointY,'bo')
            #3.2_绘制分类线
            #思路：
            # 因为模型的预测结果是通过比较sigmoid函数的输出与0.5得到的。
            # 而当sigmoid函数output为0.5时，w0 + w1*x1 + w2*x2 = 0。
            # 又因为图像的x、y轴分别为参数 x1、x2。
            # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
            x = np.linspace(-0.5,1.2,50)
            y = -(self.thetas[0][0] + self.thetas[1][0]*x)/self.thetas[2][0]
            plt.plot(x,y,'k')
            plt.pause(0.001)
            if accuracy[-1] > 0.8:
                break
            i += 1
        plt.ioff()
        plt.show()

    def SGD_getAns(self):
        """用Stochastic Gradient descent完成任务
        """
        # 初始化要用的东西（包括函数和参数等等）
        def randomError_feature(randX):
            """error * feature -> 代表着‘求导’之后的东西"""
            tempX = np.atleast_2d(self.dataX[:,randX]).T
            return (self.dataY[randX][0] - self.hypothesis(self.dataX[:,randX])) * tempX

        a = 0.003  # learning rate
        loss = [];
        step = [];
        accuracy = []  # loss图和准确率图的坐标轴
        # process & plot
        plt.ion()
        i = 0
        while 1:
            #获取随机数
            randX = random.randint(0,self.dataNum-1)
            # SGD process
            self.thetas = self.thetas + a * randomError_feature(randX)
            plt.cla()
            # 绘图,绘制 loss、准确率、拟合图
            # 1_绘制 loss 图
            plt.subplot(131)
            plt.title('loss')
            plt.xlabel("time")
            plt.ylabel("loss")
            step.append(i)  # 每幅图的x轴的刻度
            h_ans = self.hypothesis(self.dataX)  # h_ans 为traning data 的预测结果
            loss.append(self.get_loss(h_ans))  # 每幅图的y轴的刻度
            plt.plot(step, loss)
            # 2_绘制准确率图
            plt.subplot(132)
            plt.title('accuracy')
            plt.xlabel('time')
            plt.ylabel('accuracy')
            accuracy.append(self.get_accuracy(h_ans,self.dataY))
            plt.plot(step, accuracy)
            # 3_绘制分类图，由数据点和分类线构成。
            # 其中数据点用红蓝两色来区分不同的分类（该种类为trainning data 给定的）
            # 3.1_绘制数据点
            plt.subplot(133)
            plt.title('classification')
            plt.xlabel("x1")
            plt.ylabel("x2")
            redPointX = []
            redPointY = []
            bluePointX = []
            bluePointY = []
            for j in range(self.dataNum):
                if self.dataY[j] == 1:
                    redPointX.append(self.dataX[1][j])
                    redPointY.append(self.dataX[2][j])
                else:
                    bluePointX.append(self.dataX[1][j])
                    bluePointY.append(self.dataX[2][j])
            plt.plot(redPointX, redPointY, 'ro')
            plt.plot(bluePointX, bluePointY, 'bo')
            # 3.2_绘制分类线
            # 思路：
            # 因为模型的预测结果是通过比较sigmoid函数的输出与0.5得到的。
            # 而当sigmoid函数output为0.5时，w0 + w1*x1 + w2*x2 = 0。
            # 又因为图像的x、y轴分别为参数 x1、x2。
            # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
            x = np.linspace(-0.5, 1.2, 50)
            y = -(self.thetas[0][0] + self.thetas[1][0] * x) / self.thetas[2][0]
            plt.plot(x, y, 'k')
            plt.pause(0.001)
            if accuracy[-1] > 0.8:
                break
            i += 1
            plt.suptitle('logistic SGD train')
        plt.ioff()
        plt.show()

    def newton_getAns(self):
        """用牛顿迭代法求出结果
        方法：
           迭代求解，迭代过程中先求 Hessian Matrix 和 模型的一阶导数
        """
        #1_prepare
        loss = [];
        step = [];
        accuracy = []  # loss图和准确率图的坐标轴
        #2_process & plot
        plt.ion()
        i = 0
        while 1:
            plt.cla()
            #2.1_求 Hessian Matrix 和 模型的一阶导数
            hessian = np.zeros((self.dataX.shape[0],self.dataX.shape[0]))
            dj = np.zeros((self.dataX.shape[0],1))
            hAns = self.hypothesis(self.dataX)  # h_ans 为traning data 的预测结果
            for j in range(self.dataNum):
                hj = hAns[j][0]
                tempX = np.atleast_2d(self.dataX[:,j]).T
                hessian += hj * (1-hj) * tempX @ tempX.T
                dj += (hj - self.dataY[j][0]) * (np.atleast_2d(self.dataX[:,j]).T)
            hessian /= self.dataNum; dj /= self.dataNum
            self.thetas -= np.linalg.inv(hessian) @ dj
            # 绘图,绘制 loss、准确率、拟合图
            # 1_绘制 loss 图
            plt.subplot(131)
            plt.title('loss')
            plt.xlabel("time")
            plt.ylabel("loss")
            step.append(i)  # 每幅图的x轴的刻度
            hAns = self.hypothesis(self.dataX)  # h_ans 为traning data 的预测结果
            loss.append(self.get_loss(hAns))  # 每幅图的y轴的刻度
            plt.plot(step, loss, 'o')
            # 2_绘制准确率图
            plt.subplot(132)
            plt.title('accuracy')
            plt.xlabel('time')
            plt.ylabel('accuracy')
            accuracy.append(self.get_accuracy(hAns))
            plt.plot(step, accuracy,'o')
            # 3_绘制分类图，由数据点和分类线构成。
            # 其中数据点用红蓝两色来区分不同的分类（该种类为trainning data 给定的）
            # 3.1_绘制数据点
            plt.subplot(133)
            plt.title('classification')
            plt.xlabel("x1")
            plt.ylabel("x2")
            redPointX = []
            redPointY = []
            bluePointX = []
            bluePointY = []
            for j in range(self.dataNum):
                if self.dataY[j] == 1:
                    redPointX.append(self.dataX[1][j])
                    redPointY.append(self.dataX[2][j])
                else:
                    bluePointX.append(self.dataX[1][j])
                    bluePointY.append(self.dataX[2][j])
            plt.plot(redPointX, redPointY, 'ro')
            plt.plot(bluePointX, bluePointY, 'bo')
            # 3.2_绘制分类线
            # 思路：
            # 因为模型的预测结果是通过比较sigmoid函数的输出与0.5得到的。
            # 而当sigmoid函数output为0.5时，w0 + w1*x1 + w2*x2 = 0。
            # 又因为图像的x、y轴分别为参数 x1、x2。
            # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
            x = np.linspace(-0.5, 1.2, 50)
            y = -(self.thetas[0][0] + self.thetas[1][0] * x) / self.thetas[2][0]
            plt.plot(x, y, 'k')
            plt.pause(0.001)
            if self.get_accuracy(hAns) > 0.8:
                break
            i += 1
        plt.ioff()
        plt.show()