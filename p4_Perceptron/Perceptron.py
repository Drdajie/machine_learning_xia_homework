import numpy as np
import Tools.Normalization as nm
import random
import matplotlib.pyplot as plt

class Perceptron:
    """类中只有采用 SGD 方法的 Percetron"""
    def __init__(self, fileX, fileY):
        """初始化
        1_读取 training data，并处理数据。
        2_初始化参数
        :param trainDataFile: 训练数据文件名
        :param testDataFile:  测试数据文件名
        :param clfNum: classfication num
        """
        # 1_读取 training data，并处理
        # 1.1_读取
        self.dataNum, self.dataX, self.dataY = self.load_dataFile(fileX, fileY)
        # 1.2_处理
        self.xMin = self.dataX.min(axis=0)
        self.xMax = self.dataX.max(axis=0)
        self.dataX = nm.min_max_normalization(self.dataX, self.xMin, self.xMax)
        addDataX = np.ones((self.dataNum, 1))
        self.dataX = np.hstack((addDataX, self.dataX)).T
        # 2_初始化参数
        self.thetas = np.zeros((self.dataX.shape[0],))

    def load_dataFile(self,fileX,fileY):
        """读取文件
        :param dataFileName: 数据文件名
        :return: 存储数据的一个 ndarray 类型的变量，shape未做处理
        """
        dataX = np.loadtxt(fileX)
        dataY = np.loadtxt(fileY)
        num = dataX.shape[0]
        return num,dataX,dataY

    def get_loss(self,classAns):
        lossValue = 0.
        for i in range(self.dataNum):
            lossValue += (classAns[i] - self.dataY[i])\
                         * (self.thetas @ self.dataX[:,i])
        return lossValue

    def get_accuracy(self,classAns,dataY):
        """
        计算当前参数对应的预测准确率
        :param classAns: 通过 hypothesis 求得的结果，类型为一维 ndarray
        :param dataY: 真正的结果类型为一维 ndarray
        :return: 预测准确率
        """
        rightN = 0                  #rightNum
        totalN = dataY.size       #totalNum
        for i in range(totalN):
            if classAns[i] == dataY[i]:
                rightN += 1
        return rightN/totalN

    def plot_testResult(self, testXFile, testYFile=''):
        """绘制分类图，由数据点和分类线构成。
        注意:
            1_在画图之前先对数据预处理
            2_数据点用红蓝两色来区分不同的分类
        :param testXFile: 测试数据 input 的文件名
        :param testYFile: 测试数据 output 的文件名
        :return: 无
        """
        if testYFile != '':
            # 预处理
            dataNum, dataX, dataY = self.load_dataFile(testXFile, testYFile)
            dataX = nm.min_max_normalization(dataX, self.xMin, self.xMax)
            add = np.ones((dataX.shape[0], 1))
            dataX = np.hstack((add, dataX)).T
            # 1_绘制数据点
            plt.title('perceptron test')
            plt.xlabel("x1")
            plt.ylabel("x2")
            for j in range(dataX.shape[1]):
                if dataY[j] == 1:
                    plt.plot(dataX[1, j], dataX[2, j], 'ro')
                else:
                    plt.plot(dataX[1, j], dataX[2, j], 'bo')
            # 2_绘制分类线
            # 思路：
            # 超平面函数，w0 + w1*x1 + w2*x2 = 0。
            # 又因为图像的x、y轴分别为参数 x1、x2。
            # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
            x = np.linspace(0, 1, 50)
            y = -(self.thetas[0] + self.thetas[1] * x) / self.thetas[2]
            plt.plot(x, y, 'k')
            # 3_计算准确率
            classAns = self.hypothesis(dataX)
            plt.show()
            return self.get_accuracy(classAns, dataY)

    def plot_trainResult(self,i,step,loss,accuracy):
        """
        画出当前结果
        思路：
            将整张图分割成许多点，对每个点对应的坐标进行预测，并用散点图显示。
        :param i: 当前迭代回数
        :param step: 图像中显示的迭代次数
        :param loss: 图像中显示的 loss 值
        :param accuracy: 图像中显示的准确率
        :return: tempAccuracy
        """
        mk = ['+', '^', 'o']; cs = ['b', 'r', 'k']        # 分别代表散点标记、散点颜色的取值可能
        # 1_绘制 loss 图
        plt.subplot(131)
        plt.title('loss')
        plt.xlabel("time")
        plt.ylabel("loss")
        step.append(i)  # 每幅图的x轴的刻度
        classAns = self.hypothesis(self.dataX)
        loss.append(self.get_loss(classAns))  # 每幅图的y轴的刻度
        plt.plot(step, loss)
        # 2_绘制准确率图
        plt.subplot(132)
        plt.title('accuracy')
        plt.xlabel('time')
        plt.ylabel('accuracy')
        tempAccuracy = self.get_accuracy(classAns, self.dataY)
        accuracy.append(tempAccuracy)
        plt.plot(step, accuracy)
        # 3_绘制分类图
        plt.subplot(133)
        # 3.1_绘制散点
        plt.xlabel('x1')
        plt.ylabel('x2')
        for j in range(self.dataNum):
            plt.scatter(self.dataX[1, j], self.dataX[2, j],
                        marker=mk[int(self.dataY[j])],
                        c=cs[int(self.dataY[j])])
        #绘制分割线
        # 3.2_绘制分类线
        # 思路：
        # 超平面对应函数为：w0 + w1*x1 + w2*x2 = 0。
        # 又因为图像的x、y轴分别为参数 x1、x2。
        # 所以，分类线可以表示成 x2 = -(w0 + w1*x1)/w2
        x = np.linspace(0, 1, 50)
        y = -(self.thetas[0] + self.thetas[1] * x) / self.thetas[2]
        plt.plot(x, y, 'k')
        #最后的处理
        plt.suptitle('perceptron train')
        return tempAccuracy

    def hypothesis(self,data):
        """
        根据 data 得到分类结果
        :param data: input
        :return: 预测结果（如果 input 中包含多个数据，则 return 一个 list）
        """
        #1_计算 W^T * X
        pValue = self.thetas @ data
        #2_计算分类结果
        if type(pValue) == np.ndarray:
            ans = []
            for i in range(pValue.size):
                if pValue[i] >= 0:
                    ans.append(1)
                else:
                    ans.append(0)
        else:
            if pValue >= 0:
                ans = 1
            else:
                ans = 0
        return ans

    def SGD_train(self):
        """用 stochastic gradient descent 方法训练模型；显示训练过程中 loss、准确率、分类情况的变化。
        步骤：
            1_计算每种类型对应的那组参数的 error * feature
            2_更新参数
            3_画图像
        :return: 无
        """
        #初始化
        a = 0.001                                 #a为 learning rate
        step = []; accuracy = []; loss = []      #分别为迭代次数、准确率、loss大小
        def error_feature(dataIndex):
            """
            计算 error * feature
            :param dataIndex: (int) 因为是 SGD 方法，所以只含一个数,代表数据下标。
            :return: (float) error * feature 结果
            """
            tempData = self.dataX[:,dataIndex]
            return (self.dataY[dataIndex] -\
                    self.hypothesis(tempData)) * tempData
        #process
        #更新参数 & 画图
        plt.ion()
        i = 0
        while 1:
            #更新参数
            tempI = random.randint(0,self.dataNum-1)    #tempIndex
            self.thetas += a * error_feature(tempI)
            #画图
            tempA = self.plot_trainResult(i,step,loss,accuracy)
            #善后处理
            i += 1
            plt.pause(0.01)
            if tempA > 0.8:
                break
            plt.cla()
        plt.ioff()
        plt.show()