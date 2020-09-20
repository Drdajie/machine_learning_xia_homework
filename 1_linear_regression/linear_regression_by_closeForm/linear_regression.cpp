//老师课程中给的这个实验要用到的 linear model 只有两个参数，所以上手比较容易。
/*程序的步骤为：
	1.给 parameters 设置初值。
	2.自定义一个 learning rate 的初值
	3.Gradient Descent
	4.Numeric Optimization */

#include <iostream>
#include <cmath>
using namespace std;

const int PARA_NUM = 2;
const int SAMPLE_NUM = 14;

struct Site
{
	int x,y;
};

double paras[PARA_NUM];
double lr[PARA_NUM];					//learning rate

//Data
int yearX[PARA_NUM][SAMPLE_NUM];
double priceY[SAMPLE_NUM] = { 2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,7.971,8.561,10.000,11.280,12.900 };

void init()
{
	//给 parameters 设置初值，全设置为 1
	paras[0] = 1, paras[1] = -2000 + 2;

	//给 learning rate 设置初值，事实上设置多少影响都不大
	lr[0] = 0.001/2000;
	lr[1] = 0.001;

	//初始化 x 
	for (int i = 0;i < SAMPLE_NUM;i++)
	{
		yearX[0][i] = 2000 + i;
		yearX[1][i] = 1;
	}
}

double f(double ps[PARA_NUM], int x)
{
	return ps[0] * x + ps[1];
}

//使用封闭解
void close_form()
{
	//预处理
	for (int i = 0;i < PARA_NUM;i++)
		paras[i] = 0;

	//算 D = X^T * X
	double D[PARA_NUM][PARA_NUM];
	for (int i = 0;i < PARA_NUM;i++)
	{
		for (int j = 0;j < PARA_NUM;j++)
		{
			for (int k = 0;k < SAMPLE_NUM;k++)
			{
				if (k == 0)
					D[i][j] = 0;
				D[i][j] += (double)yearX[i][k] * yearX[j][k];
			}
		}
	}

	//算 D^-1
	double det = D[0][0] * D[1][1] - D[0][1]*D[1][0];
	double DD[PARA_NUM][PARA_NUM];
	DD[0][0] = D[1][1] / det;
	DD[0][1] = -D[0][1] / det;
	DD[1][0] = -D[1][0] / det;
	DD[1][1] = D[0][0] / det;

	//算 S = X^T * y
	double S[PARA_NUM][1];
	for (int i = 0;i < PARA_NUM;i++)
	{
		for (int k = 0;k < SAMPLE_NUM;k++)
		{
			if (k == 0)
				S[i][0] = 0;
			S[i][0] += yearX[i][k] * priceY[k];
		}
	}

	//算 D^-1 * S
	for (int i = 0;i < PARA_NUM;i++)
	{
		for (int j = 0;j < PARA_NUM;j++)
		{
			paras[i] += DD[i][j] * S[j][0];
		}
	}
}

int main()
{
	init();
	close_form();

	cout << "2014年预测的 price 为：                     " << f(paras,2014) << endl 
		 << "最优函数的参数 vector 的两个维度的值分别为："<< paras[0] << ',' << paras[1];
	return 0;
}
