# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:48:15 2021

@author: tremble
"""
import numpy as np
array=np.arange(1,26)
result=[]
for i in range(100):
    b=[np.random.permutation(array)]
    result.append(b)
print(result)