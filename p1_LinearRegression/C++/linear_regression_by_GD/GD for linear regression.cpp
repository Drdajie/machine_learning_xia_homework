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

double paras[PARA_NUM];
double lr;						//learning rate

//Data
int yearX[2][SAMPLE_NUM];
double priceY[SAMPLE_NUM] = { 2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,7.971,8.561,10.000,11.280,12.900 };

void init()
{
	//�� parameters ���ó�ֵ��ȫ����Ϊ 1
	paras[0] = 0.5, paras[1] = 2;

	//�� learning rate ���ó�ֵ����ʵ�����ö���Ӱ�춼����
	lr = 0.001;

	//��ʼ�� x 
	for (int i = 0;i < SAMPLE_NUM;i++)
	{
		yearX[0][i] = i;
		yearX[1][i] = 1;
	}
}

double f(double ps[PARA_NUM], int x)
{
	return ps[0] * x + ps[1];
}

double calculate_gradient(int index)
{
	double ans = 0;
	for (int i = 0;i < SAMPLE_NUM;i++)
		ans += (f(paras, yearX[0][i]) - priceY[i]) * yearX[index][i];

	return ans;
}

bool is_optimal()
{
	for (int i = 0;i < PARA_NUM;i++)
	{
		double temp = calculate_gradient(i);
		if (fabs(temp) > 1e-6)
			return false;
	}
	return true;
}

//���㵱ǰ�����¶�Ӧ�� function �� output �� ʵ�� price �ķ���
double calculate_variance(double tps[PARA_NUM])
{
	double ans = 0;
	for (int i = 0;i < SAMPLE_NUM;i++)
	{
		double temp = f(tps, yearX[0][i]) - priceY[i];
		ans += temp * temp;
	}
	return ans;
}

void numeric_optimization()
{
	double tempParas[PARA_NUM];
	for (int j = 0, k = 0;1;j = (j + 1) % 3)
	{
		//���� learning rate
		if (j == 3 && k < 3)
		{
			lr /= 2;
			k++;
		}
		//�Ż�����
		for (int i = 0;i < PARA_NUM;i++)
		{
			tempParas[i] = paras[i] - lr * calculate_gradient(i);
		}

		//cout << calculate_variance(tempParas) << endl;

		for (int i = 0;i < PARA_NUM;i++)
			paras[i] = tempParas[i];

		if (is_optimal())
			break;
	}
}

int main()
{
	init();
	numeric_optimization();

	cout << "2014��Ԥ��� price Ϊ��                     " << f(paras, 14) << endl
		 << "���ź����Ĳ��� vector ������ά�ȵ�ֵ�ֱ�Ϊ��" << paras[0] << ',' << paras[1];
	return 0;
}
