//��ʦ�γ��и������ʵ��Ҫ�õ��� linear model ֻ�������������������ֱȽ����ס�
/*����Ĳ���Ϊ��
	1.�� parameters ���ó�ֵ��
	2.�Զ���һ�� learning rate �ĳ�ֵ
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
	//�� parameters ���ó�ֵ��ȫ����Ϊ 1
	paras[0] = 1, paras[1] = -2000 + 2;

	//�� learning rate ���ó�ֵ����ʵ�����ö���Ӱ�춼����
	lr[0] = 0.001/2000;
	lr[1] = 0.001;

	//��ʼ�� x 
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

//ʹ�÷�ս�
void close_form()
{
	//Ԥ����
	for (int i = 0;i < PARA_NUM;i++)
		paras[i] = 0;

	//�� D = X^T * X
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

	//�� D^-1
	double det = D[0][0] * D[1][1] - D[0][1]*D[1][0];
	double DD[PARA_NUM][PARA_NUM];
	DD[0][0] = D[1][1] / det;
	DD[0][1] = -D[0][1] / det;
	DD[1][0] = -D[1][0] / det;
	DD[1][1] = D[0][0] / det;

	//�� S = X^T * y
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

	//�� D^-1 * S
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

	cout << "2014��Ԥ��� price Ϊ��                     " << f(paras,2014) << endl 
		 << "���ź����Ĳ��� vector ������ά�ȵ�ֵ�ֱ�Ϊ��"<< paras[0] << ',' << paras[1];
	return 0;
}
