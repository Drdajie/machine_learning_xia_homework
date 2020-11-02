import numpy as np
import Tools.Normalization as nm
import random
import matplotlib.pyplot as plt

class Muti_Perceptron:
    """类中只有采用 SGD 方法的 Percetron"""
    def __init__(self, fileX, fileY, classNum):
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
        self.classNum = classNum
        # 1.2_处理
        self.xMin = self.dataX.min(axis=0)
        self.xMax = self.dataX.max(axis=0)
        self.dataX = nm.min_max_normalization(self.dataX, self.xMin, self.xMax)
        addDataX = np.ones((self.dataNum, 1))
        self.dataX = np.hstack((addDataX, self.dataX)).T
        # 2_初始化参数
        self.thetas = np.zeros((self.dataX.shape[0],classNum))

    def load_dataFile(self,fileX,fileY):
        """读取文件
        :param dataFileName: 数据文件名
        :return: 存储数据的一个 ndarray 类型的变量，shape未做处理
        """
        dataX = np.loadtxt(fileX)
        dataY = np.loadtxt(fileY)
        num = dataX.shape[0]
        return num,dataX,dataY

    def get_loss(self):
        lossValue = 0.
        preH = self.hypothesis(self.dataX)[0]
        for i in range(self.dataNum):
            x = self.dataX[:,i]
            lossValue += preH[i] - self.thetas[:,int(self.dataY[i])] @ x
        return lossValue

    def get_accuracy(self,classAns,dataY):
        """
        计算当前参数对应的预测准确率
        :param classAns: 通过 hypothesis 求得的结果，类型为一维 ndarray
        :param dataY: 真正的结果类型为一维 ndarray
        :return: 预测准确率
        """
        rightN = 0                  #rightNum
        totalN = dataY.size         #totalNum
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
        # 初始化画图所用参数
        mk = ['+', '^', 'o']; cs = ['k', 'r', 'b'] # 分别代表背景颜色、散点标记、散点颜色的取值可能
        xL = 0; xR = 1  # 分类图 x 轴的左右边界
        yL = 0; yH = 1  # 分类图 y 轴的下上边界
        partNum = 100  # 将分类图的每个轴分多少部分
        if testYFile != '':
            dataNum, dataX, dataY = self.load_dataFile(testXFile, testYFile)
            dataX = nm.min_max_normalization(dataX, self.xMin, self.xMax)
            add = np.ones((dataNum, 1))
            dataX = np.hstack([add, dataX]).T
            # 绘制分类图
            plt.title('muti-perceptron test')
            plt.xlabel('x1');
            plt.ylabel('x2')
            # 1_绘制背景
            meshX, meshY = np.meshgrid(np.linspace(xL, xR, partNum),
                                       np.linspace(yL, yH, partNum))
            add = np.ones((meshX.size,))
            meshData = np.vstack((add, meshX.flatten()))
            meshData = np.vstack((meshData, meshY.flatten()))
            meshPrdic = self.hypothesis(meshData)[1]
            plt.contourf(meshX, meshY, meshPrdic.reshape(meshX.shape))
            # 2_绘制散点 & 打印准确率
            for i in range(dataNum):
                tempC = int(dataY[i])
                plt.scatter(dataX[1, i], dataX[2, i],
                            marker=mk[tempC], c=cs[tempC])
            plt.show()
            # 3_计算准确率
            classAns = self.hypothesis(dataX)[1]
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
        classAns = self.hypothesis(self.dataX)[1]
        loss.append(self.get_loss())  # 每幅图的y轴的刻度
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
        # 绘制背景
        # 初始化定义参数，易于之后修改
        xL = 0; xR = 1  # 分别代表散点图 x 轴的左右范围
        yL = 0; yH = 1  # 分别代表散点图 y 轴的下上范围
        tempRange = 100  # 代表要将 x、y 轴分为多少段
        meshX, meshY = np.meshgrid(np.linspace(xL, xR, tempRange),
                                   np.linspace(yL, yH, tempRange))
        # 注意 meshX、meshY 都是二维的
        tempAdd = np.ones((meshX.size,))
        meshData = np.vstack((tempAdd, meshX.flatten()))
        meshData = np.vstack((meshData, meshY.flatten()))
        meshPrdic = self.hypothesis(meshData)[1]
        plt.contourf(meshX, meshY, meshPrdic.reshape(meshX.shape))
        # 绘制散点
        plt.xlabel('x1')
        plt.ylabel('x2')
        for j in range(self.dataNum):
            plt.scatter(self.dataX[1, j], self.dataX[2, j],
                        marker=mk[int(self.dataY[j])],
                        c=cs[int(self.dataY[j])])
        plt.suptitle('muti-perceptron train')
        return tempAccuracy

    def hypothesis(self,data):
        """
        根据 data 得到分类结果
        :param data: input
        :return: 预测结果（如果 input 中包含多个数据，则 return 一个 list）
        """
        #得到数据个数
        if data.ndim == 1:
            dataNum = 1
        else:
            dataNum = data.shape[1]
        #1_计算每个 w^T * x 的结果
        tempResult = np.zeros((self.classNum,dataNum))
        for i in range(self.classNum):
            tempResult[i] = self.thetas[:,i] @ data
        #2_计算分类结果
        classAns = []
        for i in range(dataNum):
            temp_maxIndex = 0
            tempMax = 0
            for j in range(self.classNum):
                if tempMax < tempResult[j,i]:
                    temp_maxIndex = j
                    tempMax = tempResult[j,i]
            classAns.append(temp_maxIndex)
        return np.max(tempResult,axis=0),np.array(classAns)

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
        def error_feature(dataIndex,thetaIndex):
            """
            计算 error * feature
            公式为：(1(j = hw(x)) - 1(j = y)) * x
            :param dataIndex: (int) 因为是 SGD 方法，所以只含一个数,代表数据下标。
            :return: (float) error * feature 结果
            """
            x = self.dataX[:,dataIndex]
            #判断 error 部分的第一项是 0 还是 1
            if int(self.hypothesis(x)[1][0]) == thetaIndex:
                error1 = 1
            else:
                error1 = 0
            #判断 error 部分的第二项是 0 还是 1
            if int(self.dataY[dataIndex]) == thetaIndex:
                error2 = 1
            else:
                error2 = 0
            #完成公式
            return (error1 - error2) * x
        #process
        #更新参数 & 画图
        plt.ion()
        i = 0
        while 1:
            #更新参数
            tempI = random.randint(0,self.dataNum-1)    #tempIndex
            for j in range(self.classNum):
                self.thetas[:,j] -= a * error_feature(tempI,j)
            #画图
            tempA = self.plot_trainResult(i,step,loss,accuracy) #temp accuracy
            #善后处理
            i += 1
            plt.pause(0.01)
            if tempA > 0.8:
                break
            plt.cla()
        plt.ioff()
        plt.show()