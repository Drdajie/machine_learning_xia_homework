'''  This class is a father cs representing the model of linear regression, and any
   implement of it should extend it '''

import numpy as np

class Linear_Regression:
    def __init__(self):
        #This model only need two paraments.
        self.__paramentNum = 2

        #paraments -> The work f initialization leaves offspring.
        self.__theta = []
        for i in range(self.__paramentNum):
            self.__theta.append(1)

        #The training data of this homework isn't in files.
        tempGet = []
        for i in range(self.__tNum):
            tempGet.append(2000 + i)
        self.__dataX = ndarray()
        self.__dataY = np.array([2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,
                        6.853,7.971,8.561,10.000,11.280,12.900])
        
        #The term num of training data
        self.__tNum = 14                         #iNum -> term num


    def __cost_function(self,vTheta:nd,vX:nd):
        '''vTheta -> vector theta;
           vX represents dataX vector, and the every element of it also is a vector'''
        result = 0.0
        for i in range(self.__tNum):
            result += ((vTheta * vX[:,i]) - self.__dataY)**2
        return result / 2

test = Linear_Regression()
print(test.__dataY)
print(type(test.__dataY))
