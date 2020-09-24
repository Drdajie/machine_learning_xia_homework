from Linear_Regression import Linear_Regression
import matplotlib.pyplot as plt
import  Normalization as nmz
import numpy as np

class GD(Linear_Regression):
    def __init__(self):
        Linear_Regression.__init__(self)
        self.dataX[1] = nmz.min_max_normalization(self.dataX[1])
        self.dataY = nmz.min_max_normalization(self.dataY)


    def error_feature(self):                    #represent error*feature
        hs = self.hypothesis(self.dataX.T)
        myError = hs - self.dataY
        return self.dataX @ myError

    def get_ans(self,ans):
        return (12.9 - 2) * ans + 2

    def gd_process_and_show(self):
        #init paraments & picture
        itrNum = 2000                     #iterator num
        lr = 0.01                        # lr is learning rate
        plt.ion()
        lossY = []
        stepX = []

        #process
        for i in range(itrNum):
            self.thetas = self.thetas - lr * self.error_feature()
            if i < 20 or i % 100 == 0:
                #绘制 loss 图
                plt.subplot(121)
                stepX.append(i)
                lossY.append(self.cost_function())
                plt.plot(stepX,lossY)
                plt.subplot(122)
                plt.xlim(0, 1.2)
                plt.ylim(0, 14)
                plt.plot(self.dataX[1],self.get_ans(self.dataY[:,0]),'ro')
                plt.plot(self.dataX[1],self.get_ans(self.hypothesis
                                                    (self.dataX.T)[:,0]))
                plt.pause(0.001)
        plt.ioff()
        plt.show()


        #get ans
        print('gradient descent 的答案为：')
        print('最后得到的参数为：', self.thetas[:,0])
        tempX = 14/13
        ans = self.hypothesis(np.array([1, tempX]))[0]
        ans = self.get_ans(ans)
        print('2014年的预测结果为：', ans)


if __name__ == '__main__':
    gd = GD()
    gd.gd_process_and_show()