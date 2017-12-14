#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/30 15:30
# @Author  : sch
# @File    : test6.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
import pandas as pd

df = pd.read_csv('test/test6/ch06/ex1.csv')
