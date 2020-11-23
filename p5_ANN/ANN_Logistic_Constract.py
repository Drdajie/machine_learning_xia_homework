import __init__
import p5_ANN.Artificial_Neural_Network as ANN
import p2_LogisticRegression.Python.Logistic_Regression as LR
import time

trainX = '../Data/Exam/train/x.txt'
trainY = '../Data/Exam/train/y.txt'
testX = '../Data/Exam/test/x.txt'
testY = '../Data/Exam/test/y.txt'

def ann():
    #训练过程
    ann = ANN.Artificial_Neural_Network(trainX,trainY)
    ann.BP_train()
    #预测
    accuracy = ann.plot_testResult(testX,testY)
    return accuracy

def logistic():
    #训练过程
    lr = LR.Logistic_Regression(trainX,trainY)
    lr.GD_getAns()
    #预测
    accuracy = lr.plot_testResult(testX,testY)
    return accuracy

if __name__ == '__main__':
    lAccuracy = logistic()
    aAccuracy = ann()
    print('logistc model 预测准确率为：', lAccuracy)
    print('Atificial Neural Network 预测准确率为：', aAccuracy)
    time.sleep(100)