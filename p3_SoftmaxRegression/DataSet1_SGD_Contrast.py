import Softmax_Regression as SR
import p2_LogisticRegression.Python.Logistic_Regression as LR
import time
trainX = '../Data/Exam/train/x.txt'
trainY = '../Data/Exam/train/y.txt'
testX = '../Data/Exam/test/x.txt'
testY = '../Data/Exam/test/y.txt'

def softmax():
    pass
    #训练过程
    sr = SR.Softmax_Regression(trainX,trainY,2)
    sr.SGD_train()
    #预测
    accuracy = sr.plot_testResult(testX,testY)
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
    sAccuracy = softmax()
    print('logistc model 预测准确率为：', lAccuracy)
    print('softmax model 预测准确率为：', sAccuracy)
    time.sleep(100)
