import __init__
import p4_Perceptron.Muti_Perceptron as MP
import p3_SoftmaxRegression.Softmax_Regression as SR
import time
trainX = '../Data/Iris/train/x.txt'
trainY = '../Data/Iris/train/y.txt'
testX = '../Data/Iris/test/x.txt'
testY = '../Data/Iris/test/y.txt'

def muti_perceptron():
    pass
    #训练过程
    mp = MP.Muti_Perceptron(trainX,trainY,3)
    mp.SGD_train()
    #预测
    accuracy = mp.plot_testResult(testX,testY)
    return accuracy

def softmax():
    #训练过程
    sr = SR.Softmax_Regression(trainX,trainY,3)
    sr.SGD_train()
    #预测
    accuracy = sr.plot_testResult(testX,testY)
    return accuracy

if __name__ == '__main__':
    sAccuracy = softmax()
    mAccuracy = muti_perceptron()
    print('softmax model 预测准确率为：', sAccuracy)
    print('muti-perceptron model 预测准确率为：', mAccuracy)
    time.sleep(100)