import __init__
import p4_Perceptron.Perceptron as PER
import p2_LogisticRegression.Python.Logistic_Regression as LR
import time

trainX = '../Data/Exam/train/x.txt'
trainY = '../Data/Exam/train/y.txt'
testX = '../Data/Exam/test/x.txt'
testY = '../Data/Exam/test/y.txt'

def perceptron():
    pass
    #训练过程
    per = PER.Perceptron(trainX,trainY)
    per.SGD_train()
    #预测
    accuracy = per.plot_testResult(testX,testY)
    return accuracy

def logistic():
    #训练过程
    lr = LR.Logistic_Regression(trainX,trainY)
    lr.SGD_getAns()
    #预测
    accuracy = lr.plot_testResult(testX,testY)
    return accuracy

if __name__ == '__main__':
    lAccuracy = logistic()
    pAccuracy = perceptron()
    print('logistc model 预测准确率为：', lAccuracy)
    print('perceptron model 预测准确率为：', pAccuracy)
    time.sleep(100)