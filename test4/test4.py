#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27 11:11
# @Author  : sch
# @File    : test4.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# Page 112,随机漫步
import random
import numpy as np
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# 本质就是一个累计和
walk.min()
walk.max()
# 计算首次穿越时间
np.abs(walk) >= 10
(np.abs(walk) >= 10).argmax()

# 一次模拟多个随机漫步
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks

walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()
# 到达±30的数量
crossing_times = (np.abs(walks[hits30] >= 30)).argmax(1)
crossing_times.mean()

# 还可以使用的随机模型
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
