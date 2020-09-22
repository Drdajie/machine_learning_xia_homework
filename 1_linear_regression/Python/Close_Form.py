#import Linear_Regression as LR
from Linear_Regression import Linear_Regression
import numpy as np

class Close_Form(Linear_Regression):
    def __init__(self):
        Linear_Regression.__init__(self)

    def get_paraments(self):
        X = np.mat(self._Linear_Regression__dataX.T)
        y = self._Linear_Regression__dataY
        self.thetas = (X.T @ X).I @ X.T @ y
        return self.thetas
    

def show_result():
    cf = Close_Form()
    paras = np.array(cf.get_paraments())[0]
    print('最后得到的参数为：',paras)
    print('2014年的预测结果为：',cf.hypothesis(paras,np.array([1,2014])))

show_result()
