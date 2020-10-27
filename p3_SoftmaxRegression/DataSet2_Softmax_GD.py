import Softmax_Regression as SR
import time
trainX = '../Data/Iris/train/x.txt'
trainY = '../Data/Iris/train/y.txt'
testX = '../Data/Iris/test/x.txt'
testY = '../Data/Iris/test/y.txt'

def softmax():
    pass
    #训练过程
    sr = SR.Softmax_Regression(trainX,trainY,3)
    sr.GD_train()
    #预测
    accuracy = sr.plot_testResult(testX,testY)
    return accuracy

if __name__ == '__main__':
    accuracy = softmax()
    print('用 GD 方法实现 softmax model 得到的预测准确率为:',accuracy)
    time.sleep(100)