import matplotlib.pyplot as plt
from numpy import *
import numpy as np

# Logistic回归梯度上升优化算法
# 读取数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('../Data/ex4Data/ex4x.dat')
    for line in fr.readlines():
        lineArr = line.strip().split('  ')
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])

    fr = open('../Data/ex4Data/ex4y.dat')
    for line in fr.readlines():
        labelMat.append(float(line.strip()))

    dataMatrix = mat(dataMat)

    # Z-score 归一化
    bias, dataMatrix = np.split(dataMatrix, [1], axis=1)
    myMean = dataMatrix.mean(axis=0)
    myStd = dataMatrix.std(axis=0)         #standard deviation
    '''myMax = dataMatrix.max(axis = 0)
    myMin = dataMatrix.min(axis = 0)        '''
    dataMatrix = (dataMatrix - myMean) / myStd
    dataMatrix = np.hstack((bias, dataMatrix))

    dataMat = dataMatrix.tolist()

    return dataMat, labelMat


# 定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def calculate_loss(h,y,num):
    loss = 0
    for i in range(num):
        tempH,tempY = h[i,0],y[i,0]
        loss += math.pow(tempH,tempY) * math.pow(1-tempH,1-tempY)
    return loss

# 输入数据特征与数据的类别标签
# 返回最佳回归系数(weights)
#改进的随机梯度上升法（随机化）
#使用随机的一个样本来更新回归系数
#numIter迭代
def stoGradAscent(dataMatrix, classLabels, numIter = 5000):
    #m为行 n为列
    m,n = shape(dataMatrix)
    labelMat = mat(classLabels).transpose()
    weights = ones((n,1))
    alpha = 0.001
    lossValues = []
    # 随机梯度, 循环150,观察是否收敛
    for j in range (numIter):
        h = sigmoid(dataMatrix * weights)
        # [0, 1, 2 .. m-1]
        dataIndex = range(m)
        randIndex = int(random.uniform(0, len(dataIndex)))
        #随机选取更新
        error = classLabels[randIndex] - h[randIndex,0]
        weights = weights + alpha * mat(dataMatrix[randIndex]).transpose() * error
        del (list(dataIndex)[randIndex])
        lossValues.append(calculate_loss(h, labelMat, m))
    return array(weights),lossValues,numIter

def plotBestFit(weights,lossValues,numIter):
    #第一个figure
    fig = plt.figure()
    bx = fig.add_subplot(111)
    bx.scatter(range(numIter), lossValues, s=30)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.show()

    #第二个figure
    dataMat, labelMat = loadDataSet()

    dataArr = array(dataMat)
    # n->数据量，样本数
    n = shape(dataArr)[0]
    # xcord1,ycord1代表正例特征
    # xcord2,ycord2代表负例特征
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []

    # 循环筛选出正负集 & 统计正确数
    dataMatrix = mat(dataMat)
    h = sigmoid(dataMatrix * weights)
    rightNum = 0
    for i in range(n):
        if h[i,0] > 0.5:
            tempC = 1
        else:
            tempC = 0
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
        if int(labelMat[i] == tempC):
            rightNum += 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 设定边界直线x和y的值
    x = arange(-3.0, 3.0, 0.1)
    """
    y的由来？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1， x2就是我们画图的y值。0是两个类别的分界处
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('正确率：', rightNum / n)

dataMat, labelMat = loadDataSet()
print('最后得到的参数为：')
weights,lossValus,numIter = stoGradAscent(mat(dataMat), labelMat)
print(weights)
plotBestFit(weights,lossValus,numIter)

