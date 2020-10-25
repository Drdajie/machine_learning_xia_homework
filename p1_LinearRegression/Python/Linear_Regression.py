'''This modle only has one class named Linear_Regression as parent class '''
import numpy as np

class Linear_Regression:
    '''This class is a father cs representing the model of linear regression, and any
       implement of it should extend it '''
    def __init__(self):
        #This model only need two paraments.
        self.paramentNum = 2
        #paraments -> The work f initialization leaves offspring.
        self.thetas = np.ones((1,self.paramentNum)).T
        #The term num of training data
        self.tNum = 14                         #iNum -> term num
        #The training data of this homework isn't in files.
        #All vector datas are set as column vector.
        self.dataX = np.ones((self.paramentNum,self.tNum))
        for i in range(self.tNum):
            self.dataX[1,i] = 2000 + i
        self.dataY = np.array([2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,        #the type of dataY is list
                        6.853,7.971,8.561,10.000,11.280,12.900])
        self.dataY = np.atleast_2d(self.dataY).T
        #get the min、max values of dataX and dataY
        self.xMin = 2000;self.xMax = 2013
        self.yMin = 2.;self.yMax = 12.9

    #get the prediction result
    def hypothesis(self,vX):                    #Both theta and x are vector
        return (np.dot(vX,self.thetas))


    def cost_function(self,tempX,tempY):
        '''vTheta -> vector theta;
           vX represents dataX vector, and the every element of it also is a vector'''
        tempError = self.hypothesis(tempX.T) - tempY
        result = np.dot(tempError.T,tempError)[0,0]
        return result / 2
