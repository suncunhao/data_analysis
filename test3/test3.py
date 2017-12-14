#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 17:41
# @Author  : sch
# @File    : test3.py
# blog:http://www.cnblogs.com/batteryhp/p/5000104.html

import numpy as np

data = [[1,2,5.6],[21,4,2]]
# print type(data)
# 此时类型为list
# print data
# print data*10 对list为重复十次
# print data+data
data = np.array(data)
# 列表转化为数组
# print type(data)
# 此时类型为ndarray
# print data*10 对数组为元素乘以十倍
# print data.shape
# 数组大小
# print data.dtype
# 数组数据类型
# print data.ndim
# 数组维度

# a = np.zeros(10)
# print a
# 创建零数组
# b = np.ones((3,6))
# print b
# 创建一个全是1元素的数组
# c = np.empty((2,3,2))
# print c
# 创建一个没有具体值的数组：empty

# d = np.arange(15)
# print d
# arange函数：range函数的数组版，与range的区别就是创建的是数组而不是列表

# 花式索引
arr = np.empty((8,4))
for i in range(arr.shape[0]):
    # 在这里注意shape的用法，后面不加括号
    arr[i] = i
print arr
print arr[[3,5,4,2]]
print arr[[-2,-1,-4]]
arr2 = np.arange(32).reshape((8,4))
print arr2
print arr2[[1,5,7,2],[0,3,1,2]]
print arr2[[1,5,7,2]][:,[0,3,1,2]]
print arr2[np.ix_([1,5,7,2],[0,3,1,2])]
# 上面的按索引提数据，倒数二三行的输出结果有区别应当注意

