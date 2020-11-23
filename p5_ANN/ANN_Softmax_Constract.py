import __init__
import p5_ANN.Artificial_Neural_Network as ANN
import p3_SoftmaxRegression.Softmax_Regression as SR
import time
trainX = '../Data/Iris/train/x.txt'
trainY = '../Data/Iris/train/y.txt'
testX = '../Data/Iris/test/x.txt'
testY = '../Data/Iris/test/y.txt'

def ann():
    #训练过程
    ann = ANN.Artificial_Neural_Network(trainX,trainY)
    ann.BP_train()
    #预测
    accuracy = ann.plot_testResult(testX,testY)
    return accuracy

def softmax():
    #训练过程
    sr = SR.Softmax_Regression(trainX,trainY,3)
    sr.GD_train()
    #预测
    accuracy = sr.plot_testResult(testX,testY)
    return accuracy

if __name__ == '__main__':
    sAccuracy = softmax()
    aAccuracy = ann()
    print('softmax model 预测准确率为：', sAccuracy)
    print('ann 预测准确率为：', aAccuracy)
    time.sleep(100)