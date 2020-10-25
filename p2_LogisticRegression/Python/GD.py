import Logistic_Regression as LR
fileX = '../../Data/ex4Data/ex4x.dat'
fileY = '../../Data/ex4Data/ex4y.dat'

if __name__ == "__main__":
    def get_ans():
        lr = LR.Logistic_Regression(fileX,fileY)
        lr.GD_getAns()
    get_ans()
