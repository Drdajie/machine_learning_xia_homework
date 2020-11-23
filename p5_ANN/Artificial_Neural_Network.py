import numpy as np
import sys
sys.path.append('../')
import Tools.Normalization as nm
import Tools.Common_Function as cf
import matplotlib.pyplot as plt

class Artificial_Neural_Network:
    """三层神经网络模型实现"""
    def __init__(self, fileX, fileY):
        """
        初始化
        1_读取 training data，并处理数据。
        2_初始化参数
        :param trainDataFile: 训练数据文件名
        :param testDataFile:  测试数据文件名
        """
        # 1_读取 training data，并处理
        # 1.1_读取
        self.dataNum, self.dataX, dataY = self.load_dataFile(fileX, fileY)
        self.inputNum = self.dataX.shape[1]
        self.outputNum = int(np.max(dataY) + 1)
        self.hiddenNode_num = 5
        # 1.2_处理
        self.xMin = self.dataX.min(axis=0)
        self.xMax = self.dataX.max(axis=0)
        self.dataX = nm.min_max_normalization(self.dataX, self.xMin, self.xMax)
        self.dataX = self.dataX.T
        self.dataY = np.zeros((self.outputNum,self.dataNum))
        for i in range(dataY.size):
            self.dataY[int(dataY[i]),i] = 1
        # 2_初始化参数 -> 默认 hidden 层有 5 个神经元
        self.w_ih = np.random.random((self.inputNum,self.hiddenNode_num))
        self.b_h = np.random.random((self.hiddenNode_num,))
        self.w_ho = np.random.random((self.hiddenNode_num,self.outputNum))
        self.t_o = np.random.random((self.outputNum,))                       #theta_output

    def load_dataFile(self,fileX,fileY):
        """读取文件
        :param dataFileName: 数据文件名
        :return: 存储数据的一个 ndarray 类型的变量，shape未做处理
        """
        dataX = np.loadtxt(fileX)
        dataY = np.loadtxt(fileY)
        num = dataX.shape[0]
        return num,dataX,dataY

    def get_cost(self,yPre,yReal):
        """
        计算 cost
        :param yPre: y的预测结果。类型为向量或矩阵
        :param yReal：y的真实值
        :return: 值或者向量
        """
        if yPre.ndim == 1:
            ans = 0.
            for i in range(yPre.size):
                ans += (yPre[i] - yReal[i])**2
            ans /= 2
        else:
            ans = np.zeros((yPre.shape[1],))
            for j in range(yPre.shape[1]):
                for i in range(yPre.shape[0]):
                    ans[j] += (yPre[i,j] - yReal[i,j])**2
                ans[j] /= 2
        return np.sum(ans)

    def calculateY(self,data):
        """
        计算给出的 data 的预测结果
        :param data: 向量或向量组
        :return: 预测结果和中间层的中间结果
        """
        if data.ndim == 1:
            ah = data @ self.w_ih           #as.shape = (self.hiddenNode_num,)
            bh = cf.sigmoid(ah + self.b_h)
            bt = bh @ self.w_ho             #beta
            ys = cf.sigmoid(bt + self.t_o)  #ys.shape = (self.outputNum,)
        else:
            ah = (data.T @ self.w_ih)
            bh = cf.sigmoid(ah + self.b_h)
            bt = bh @ self.w_ho  # beta
            ys = cf.sigmoid(bt + self.t_o)
            bh = bh.T; ys = ys.T
        return bh,ys

    def get_preClass(self,yPre):
        if yPre.ndim == 1:
            oneIndex = np.argmax(yPre)              # 1 所在的位置
            y_preC = np.zeros(yPre.shape)           #y_predict class
            y_preC[oneIndex] = 1
        else:
            oneIndex = np.argmax(yPre,axis=0)
            y_preC = np.zeros(yPre.shape)
            for j in range(yPre.shape[1]):
                y_preC[oneIndex[j]] = 1
        return oneIndex,y_preC

    def get_accuracy(self,yPre,yReal):
        up = 0
        under = yReal.shape[1]
        for i in range(under):
            tempY = np.zeros((self.outputNum,))
            maxIndex = 0
            tempMax = 0
            for j in range(yPre.shape[0]):
                if tempMax < yPre[j,i]:
                    tempMax = yPre[j,i]
                    maxIndex = j
            tempY[maxIndex] = 1
            if (tempY == yReal[:,i]).all():
                up += 1
        return up/under

    def plot_trainResult(self, i, step, loss, accuracy):
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
        mk = ['+', '^', 'o'];  cs = ['b', 'r', 'k']  # 分别代表散点标记、散点颜色的取值可能
        # 1_绘制 loss 图
        plt.subplot(131)
        plt.title('loss')
        plt.xlabel("time")
        plt.ylabel("loss")
        step.append(i)  # 每幅图的x轴的刻度
        yPre = self.calculateY(self.dataX)[1]
        loss.append(self.get_cost(yPre,self.dataY))  # 每幅图的y轴的刻度
        plt.plot(step, loss)
        # 2_绘制准确率图
        plt.subplot(132)
        plt.title('accuracy')
        plt.xlabel('time')
        plt.ylabel('accuracy')
        tempAccuracy = self.get_accuracy(yPre, self.dataY)
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
        meshData = np.vstack((meshX.flatten(), meshY.flatten()))
        meshPredic = self.calculateY(meshData)[1]
        meshClass = self.get_preClass(meshPredic)[0]
        plt.contourf(meshX, meshY, meshClass.reshape(meshX.shape))
        # 绘制散点
        plt.xlabel('x1')
        plt.ylabel('x2')
        pltTool = np.argmax(self.dataY,axis=0)
        for j in range(self.dataNum):
            plt.scatter(self.dataX[0, j], self.dataX[1, j],
                        marker=mk[pltTool[j]], c=cs[pltTool[j]])
        plt.suptitle('ANN train')
        return tempAccuracy

    def BP_train(self):
        #初始化
        a = 0.03                   #learning rate
        step = []; accuracy = []; loss = []      #分别为迭代次数、准确率、loss大小
        #process
        #更新参数 & 画图
        plt.ion()
        i = 0
        while 1:
            #更新参数
            for j in range(self.dataNum):
                tempB,tempY = self.calculateY(self.dataX[:,j])
                error_output = (tempY - self.dataY[:,j]) * tempY * (1-tempY)       #error^OutputLayer
                error_hidden = tempB * (1-tempB) * (error_output @ self.w_ho.T)    #error^HiddenLayer
                dW = tempB.reshape(-1,1) @ error_output.reshape(1,-1)
                dT = error_output                                                  #dTheta
                dV = self.dataX[:,j].reshape(-1,1) @ error_hidden.reshape(1,-1)
                dG = error_hidden                                                  #dGarma
                self.w_ho -= a * dW
                self.t_o -= a * dT
                self.w_ih -= a * dV
                self.b_h -= a * dG
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
            dataNum, dataX, tempDataY = self.load_dataFile(testXFile, testYFile)
            dataX = nm.min_max_normalization(dataX, self.xMin, self.xMax)
            dataX = dataX.T
            dataY = np.zeros((self.outputNum, dataNum))
            for i in range(tempDataY.size):
                dataY[int(tempDataY[i]), i] = 1
            xL = 0; xR = 1              # 分别代表散点图 x 轴的左右范围
            yL = 0; yH = 1              # 分别代表散点图 y 轴的下上范围
            mk = ['+', '^', 'o'];  cs = ['b', 'r', 'k']  # 分别代表散点标记、散点颜色的取值可能
            plt.title('ann test')
            plt.xlabel("x1")
            plt.ylabel("x2")
            # 1_绘图
            # 1.1_绘制背景
            tempRange = 100  # 代表要将 x、y 轴分为多少段
            meshX, meshY = np.meshgrid(np.linspace(xL, xR, tempRange),
                                       np.linspace(yL, yH, tempRange))
            # 注意 meshX、meshY 都是二维的
            meshData = np.vstack((meshX.flatten(), meshY.flatten()))
            meshPredic = self.calculateY(meshData)[1]
            meshClass = self.get_preClass(meshPredic)[0]
            plt.contourf(meshX, meshY, meshClass.reshape(meshX.shape))
            # 1.2_绘制散点
            pltTool = np.argmax(dataY, axis=0)
            for j in range(dataNum):
                plt.scatter(dataX[0, j], dataX[1, j],
                            marker=mk[pltTool[j]], c=cs[pltTool[j]])
            plt.show()
            # 2_计算准确率
            yPre = self.calculateY(dataX)[1]
            accuracy = self.get_accuracy(yPre, dataY)
            return accuracy