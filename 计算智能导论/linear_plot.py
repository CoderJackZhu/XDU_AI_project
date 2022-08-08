# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:04:00 2021

@author: tremble
"""

import  matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
x1 = [3,6]
x2 = [4,5]
x3 = [0,0]
plt.plot(x1[0],x1[1],'*')
plt.plot(x2[0],x2[1],'*')
for x in range(0,201):
	for y in range(0,201):
		i = 0.01 * x
		j = 0.01 * y
		if(i+j == 2 or i==0 or j==0):
			x3[0] = i*x1[0] + j*x2[0]
			x3[1] = i*x1[1] + j*x2[1]
			plt.plot(x3[0],x3[1],'.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('线性')
plt.show()
