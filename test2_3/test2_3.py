#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 10:03
# @Author  : sch
# @File    : test2_3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names1880 = pd.read_csv('names/yob1880.txt', names=['name', 'sex', 'birth'])
# print names1880
names1880_grouped = names1880.groupby('sex').birth.sum()
# print names1880_grouped

# 由于数据集被分隔成了多个文件，所以先合并为一个DataFrame，加上year字段
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'names/yob%d.txt' % year
    # 字符串格式化
    frame = pd.read_csv(path, names=columns)
    # 对数据进行读取，单个frame就像names1880一样的形式
    frame['year'] = year
    # print frame
    # 对frame数据加入years列
    pieces.append(frame)
    # 将frame数据加入pieces列表
    # 注意，此时pieces的形式为list，如[A,B,C,D...]这样形式
names = pd.concat(pieces, ignore_index=True)
# concat:将数据根据不同的轴作简单融合，注意处理对象为DataFrame
# ignore_index=True参数作用：不保留read_csv返回的原始行号
# print np.size(pieces)
# print names
'''
print type(pieces)
print type(names)
可以看出二者的类型不一样
'''

# 得到了这个数据之后，就可以使用groupby和pivot_table函数对感兴趣的year与sex进行分析了
total_births = names.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
# 数据透视表的运用：birth、
# print total_births.tail()
# total_births.plot(title='Total births by sex and year')
# plt.show()

# 下面来插入prop列，用于存放指定名称的婴儿数相对于总出生数的比例
#p rop=0.02，说明100名婴儿中有2个取了当前名字
def add_prop(group):
    # 注意这里是python2的整数除法
    # 将数据转换为float类型（为什么需要转换为float）:由于birth为整数，必须在计算的时候将分子或者分母转换为浮点数
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    # .births的调用方法
    return group
# print type(names)
names = names.groupby(['year', 'sex']).apply(add_prop)
# print names
# names.plot('year',title='births of year')
# plt.show()
# 接下来有效性检查：prop分组的和是否近似为1，返回为True
# print np.allclose(names.groupby(['year','sex']).prop.sum(),1)

# get_top函数的具体运用是什么：输出每年不同性别的前n个姓名
def get_top(group, n):
    return group.sort_values(by='births', ascending=False)[:n]
grouped = names.groupby(['year', 'sex'])
# print grouped
# 变量grouped是一个Groupby对象，它实际上还没有进行任何计算
# 相当于：将names按照year以及sex归类，然后再进行get_top1000操作
top1000 = grouped.apply(get_top, 1000)
# print top1000

# 接下来根据top1000，我们可以分析命名趋势
# 根据男女分组
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
# 做出了两个简单的时间序列:top1000是按时间排的
total_births = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)
# 这步会报错，不知道为什么
# total_births = total_births.fillna(0)
# 观察发现数据中有NaN项，可以选择填充
# print total_births
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn', 'Aaron']]
# subset.plot(subplots=True, figsize=(12,10), grid=False, title='Nmuber of births per year')
# plt.show()
# plot函数的参数含义
# subplots是用来标明是否将几个图画在一起, figsize用来标明大小，grid 是标明是否有网格线

# 评估命名多样性的增长
# table = top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
# table.plot(title='Sum of prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
# plt.show()
# 可以看出名字的多样性出现了增长：前1000项的比例降低
# 另一个方法是计算出占总出生人数前50%的不同名字的数量
# 这个数字不好计算，仅考虑2010年男孩的名字：（并且对prop是降序排列的）
df = boys[boys.year == 2010]
# print df
# 然后我们可以看，需要多少个prop求和能到50%
# for可以做到，但是使用numpy的累计求和cumsum，然后通过searchsorted寻找0.5插入的位置
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
# print prop_cumsum[:10]
# print prop_cumsum.searchsorted(0.5)
# 输出为116+1=117（因为从0开始计数），与1900年的数据比这个数值大了很多
df2 = boys[boys.year == 1900]
in1900 = df.sort_values(by='prop', ascending=False).prop.cumsum()
in1900.searchsorted(0.5)+1
# 这个结果是25

# 接下来我们可以对所有结果进行这个计算了
# 按year与sex字段进行groupby处理，然后用一个函数计算各个分组的这个值
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q)[0]+1
# 注意！！！这里和书本不一样，上面看到python3的searchsorted()返回的是ndarray类型
# 需要先取[0]元素，才能获得想要的数据，如果不作该处理，绘图会报错

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
# 依靠sex入栈操作，变Series为DataFrame
# print diversity.head()
# diversity.plot(title='Number of popluar names in top 50%')
# plt.show()

# 下面看最后一个字母的变革
# 首先从name列取出最后一个字母，这里使用了lamda创建匿名函数
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
# map函数的用法：接收一个函数f和list，将f依次作用在list上，得到一个新的list
# print last_letters
# print type(last_letters)
last_letters.name = 'last_latter'
# print last_letters
# 这一步不知道在做什么，有可能是为了下面的数据透视表准备的
table = names.pivot_table('births', index=last_letters, columns=['sex','year'],aggfunc=sum)
# 为什么names表中没有last_letters但是可以在index使用
# print table
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
# print subtable.head()
# 进行规范化处理，以便计算各性别各末字母占总出生人数的比例
# print subtable.sum()
letter_prop = subtable / subtable.sum().astype(float)
# print letter_prop
# fig,axes = plt.subplots(2,1,figsize=(10,8))
# letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='Male')
# letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='Female',legend=False)
# plt.show()
# 可以得到各末字母所占比例

# 接下来按年度与性别进行规范化处理，并在男孩姓名中选取几个字母，进行转置以便将各个列做成一个时间序列：
letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y', 'b'], 'M'].T
# print dny_ts.head()
# dny_ts.plot()
# plt.show()
# 绘制一张趋势图

# 查看姓名颠倒情况，如女名变男名等
all_names = top1000.name.unique()
# unique函数：去重
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
# print lesley_like
# 找出了所有含有lesl的姓名
# 接下来过滤其他名字，并按名字分组计算出生数，查看相对频率
filtered = top1000[top1000.name.isin(lesley_like)]
# isin函数和where等的区别
# print filtered
# print filtered.groupby('name').births.sum()
table = filtered.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
# print table
table = table.div(table.sum(1), axis=0)
# div函数：做归一化
# print table
# print table.head()
# print table.tail()
table.plot(style={'M': 'k-', 'F': 'k--'})
plt.show()
# 画出反转趋势图