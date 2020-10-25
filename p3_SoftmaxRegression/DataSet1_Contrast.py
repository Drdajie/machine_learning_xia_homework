import Softmax_Regression as SR
import p2_LogisticRegression.Python.Logistic_Regression as LR
import p3_SoftmaxRegression.Softmax_Regression as SR
trainX = '../Data/Exam/train/x.txt'
trainY = '../Data/Exam/train/y.txt'
testX = '../Data/Exam/test/x.txt'
testY = '../Data/Exam/test/y.txt'

def softmax():
    pass
    #训练过程
    #sr =
    # sr.GD_getAns()
    #预测

def logistic():
    lr = LR.Logistic_Regression(trainX,trainY)
    lr.GD_getAns()

if __name__ == '__main__':
    logistic()