#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/1 14:39
# @Author  : sch
# @File    : test8.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.plot(np.arange(10))
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

from numpy.random import randn
plt.plot(randn(50).cumsum(), 'k--')
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
_ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
fig, axes = plt.subplots(2, 3)
axes

# 调整subplot周围的间距
# subplots_adjust(left=None, bottom=None, right=None, top=None,
#               wspace=None, haspac=None)
# wspace和hspace用于控制宽度和高度的百分比，可以用作subplot之间的间距
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)

# 颜色、标记和线型
plt.plot(randn(30).cumsum(), 'ko--')
plt.plot(randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
data = randn(30).cumsum()
plt.plot(data, 'k--', linestyle='dashed', marker='o')
plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
plt.legend(loc='best')

# 刻度、标签和图例
# 设置标题、轴标签、刻度以及刻度标签
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')

# 添加图例
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k', label='one')
ax.plot(randn(1000).cumsum(), 'k--', label='two')
ax.plot(randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')

# 注解以及在subplot上绘图
# 注解可以通过text、arrow和annotate等函数进行添加，text可以将文本
# 绘制在图表的指定坐标(x, y)，还可以加上一些自定义格式
# ax.text(x, y, 'Hello world!',
#         famliy='monospace', fontsize=10)

# from datetime import datetime
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# data = pd.read_csv('test/test8/spx.csv')
# spx = data['SPX']
# spx.plot(ax=ax, style='k-')
# crisis_data = [
#     (datetime(2007, 10, 11), 'Peak of bull market'),
#     (datetime(2008, 3, 12), 'Bear Stearns Fails'),
#     (datetime(2008, 9, 15), 'Lehman Bankruptcy')
# ]
# for date, label in crisis_data:
#     ax.annotate(label, xy=(date, spx.asof(date) + 50),
#                 xytext=(date, spx.asof(date) + 200),
#                 arrowprops=dict(facecolor='black'),
#                 horizontalalignment='left', verticalalignment='top')
#     # 放大到2007-2010
# ax.set_xlim(['1/1/2007', '1/1/2011'])
# ax.set_ylim([600, 1800])
# ax.set_title('Import dates in 2008-2009 financial crisis')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

# 将图表保存到文件
# plt.savefig将当前图表保存到文件
# plt.savefig('figpath.png', dpi=400, bbox_inches='tight')

# matplotlib配置
# plt.rc('figure', figsize=(10, 10))
'''
font_options = {'famliy': 'monospace',
                'weight': 'bold',
                'size': 'small'}
plt.rc('font', **font_options)
'''

# pandas中的绘图函数
# 线型图
s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
                  columns=['A', 'B', 'C', 'D'],
                  index=np.arange(0, 100, 10))
df.plot()
'''
Series.plot方法的参数
label   图例的标签
ax      要在其上进行绘制的matplotlib.subplot图像
style   将要传给matplotlib的风格字符串(如'ko--')
alpha   图表的填充不透明度
kind    可以是line,bar,barh,kde
logy    在Y轴上使用对数标尺
use_index   将对象的索引用作刻度标签
rot     旋转刻度标签
xticks  X轴刻度的值
yticks  Y轴刻度的值
xlim    X轴界限
ylim    Y轴界限
grid    显示轴网格线
'''
'''
DataFrame的plot参数
subplots        将各个DataFrame列绘制到单独的subplot中
sharex          如果subplot=True，则共用一个X轴
sharey          如果subplot=True，则共用一个Y轴
figsize         绘图大小
title           绘图标题
legend          添加一个subplot图例
sort_columns    以字母表绘制各列
'''

# 柱状图
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
data.plot(kind='bar', ax=axes[1], color='k', alpha=0.7)

df = pd.DataFrame(np.random.rand(6, 4),
                  index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D']))
df
df.plot(kind='bar')
df.plot(kind='barh', stacked=True, alpha=0.5)

# tips = pd.read_csv('test/test8/tips.csv')
# party_counts = pd.crosstab(tips.day, tips.size)
# party_counts = party_counts.iloc[:, 2: 5]
# party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
# party_pcts
# party_pcts.plot(kind='bar', stacked=True)

# 直方图和密度图
# 直方图：hist(bins=50)
# 密度图：plot(kind='kde')
comp1 = np.random.normal(0, 1, size=200)
comp2 = np.random.normal(10, 2, size=200)
values = pd.Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k')
# concatenate函数

# 散布图
macro = pd.read_csv('test/test8/macrodata.csv')
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
trans_data[-5:]
plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))
pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)

# 绘制地图：图形化显示海地地震危机数据
data = pd.read_csv('test/test8/Haiti.csv')
data
data[['INCIDENT DATE', 'LATITUDE', 'LONGITUDE']][:10]
data['CATEGORY'][:6]
data.describe()
data = data[(data['LATITUDE'] > 18) & (data['LATITUDE'] < 20) &
            (data['LONGITUDE'] > -75) & (data['LONGITUDE'] < -70) &
            data['CATEGORY'].notnull()]
def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]
def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))
def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return code, names.strip()
get_english('2. Urgences logistiques | Vital Lines')

all_cats = get_all_categories(data.CATEGORY)
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping['2a']
english_mapping['6c']

def get_code(seq):
    return [x.split('.')[0] for x in seq if x]
all_codes = get_code(all_cats)
code_index = pd.Index(np.unique(all_codes))
dummy_frame = pd.DataFrame(np.zeros((len(data),len(code_index))),
                           index=data.index, columns=code_index)
dummy_frame.iloc[:, :6]
for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.loc[row, codes] = 1
data = data.join(dummy_frame.add_prefix('category_'))
