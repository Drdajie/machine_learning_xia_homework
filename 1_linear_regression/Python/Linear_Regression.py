'''This modle only has one class named Linear_Regression as parent class '''
import numpy as np

'''  This class is a father cs representing the model of linear regression, and any
   implement of it should extend it '''
class Linear_Regression:
    def __init__(self):
        #This model only need two paraments.
        self.__paramentNum = 2

        #paraments -> The work f initialization leaves offspring.
        self.thetas = np.ones((1,self.__paramentNum))

        #The term num of training data
        self.__tNum = 14                         #iNum -> term num
        
        #The training data of this homework isn't in files.
        #All vector datas are set as column vector.
        self.__dataX = np.ones((self.__paramentNum,self.__tNum))
        for i in range(self.__tNum):
            self.__dataX[1,i] = 2000 + i
        self.__dataY = np.array([2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,        #the type of dataY is list
                        6.853,7.971,8.561,10.000,11.280,12.900]).T


    #get the prediction result
    def hypothesis(self,vTheta,vX):                    #Both theta and x are vector
        return (np.dot(vTheta,vX))
    

    def cost_function(self,vTheta,vXs):
        '''vTheta -> vector theta;
           vX represents dataX vector, and the every element of it also is a vector'''
        result = 0.0
        for i in range(self.__tNum):
            result += (hypothesis(vTheta,vXs[:,i]) - self.__dataY[i])**2
        return result / 2
