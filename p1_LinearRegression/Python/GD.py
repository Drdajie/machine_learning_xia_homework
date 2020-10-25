from Linear_Regression import Linear_Regression
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
import Tools.Normalization as nm
import numpy as np
import time
import copy

class GD(Linear_Regression):
    def __init__(self):
        '''to copy the samples and make normalization'''
        Linear_Regression.__init__(self)
        #to copy the samples
        self.tempX = copy.copy(self.dataX)
        self.tempY = copy.copy(self.dataY)
        #normalization
        self.tempX[1] = nm.min_max_normalization(self.tempX[1],self.xMin,self.xMax)
        self.tempY = nm.min_max_normalization(self.tempY,self.yMin,self.yMax)


    def error_feature(self):
        """error * feature"""
        hs = self.hypothesis(self.tempX.T)
        myError = hs - self.tempY
        return self.tempX @ myError


    def gd_process_and_show(self):
        #init paraments & picture
        itrNum = 50                       #iterator num
        lr = 0.01                         # lr is learning rate
        #plt.ion()
        plt.ioff()
        lossY = []
        stepX = []

        #process
        for i in range(itrNum):
            self.thetas = self.thetas - lr * self.error_feature()
            plt.cla()
            #绘制 loss 图
            plt.subplot(121)
            stepX.append(i)
            lossY.append(self.cost_function(self.tempX,self.tempY))
            plt.plot(stepX,lossY)
            #绘制拟合图
            plt.subplot(122)
            plt.xlim(self.xMin,self.xMax + 1)
            plt.ylim(self.yMin,self.yMax)
            plt.plot(self.dataX[1],
                      nm.reverse_mm_normalization(self.tempY[:,0],self.yMin,self.yMax),'ro')
            plt.plot(self.dataX[1],
                      nm.reverse_mm_normalization(self.hypothesis(self.tempX.T)[:,0],
                                                   self.yMin,self.yMax))
            plt.pause(0.001)
        plt.ioff()
        plt.show()


        #get ans
        print('gradient descent 的答案为：')
        print('最后得到的参数为：', self.thetas[:,0])
        tempX = 14/13
        ans = self.hypothesis(np.array([1, tempX]))[0]
        ans = nm.reverse_mm_normalization(ans,self.yMin,self.yMax)
        print('2014年的预测结果为：', ans)

if __name__ == '__main__':
    gd = GD()
    gd.gd_process_and_show()
