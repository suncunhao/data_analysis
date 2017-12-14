#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/18 9:23
# @Author  : sch
# @File    : test2_1.py

# 引用模块
import json
from collections import defaultdict
from collections import  Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'usagov_bitly_data2012-03-16-1331923249.txt'

records = [json.loads(s) for s in open(path)]

# print records[0]
# print records[0]['tz']

# time_zone = [rec['tz'] for rec in records]
time_zone = [rec['tz'] for rec in records if 'tz' in rec]
# print time_zone
'''
另一种写法
time_zone = []
for rec in records:
    if 'tz' in rec:
        time_zone.append(rec['tz'])
print time_zone
'''
# print time_zone[:10]
'''
def get_counts(sequence):
    counts = {} #创建空字典
    for x in sequence:
        if x in counts:
            counts[x]+= 1
        else:
            counts[x] = 1
    return counts
counts = get_counts(time_zone)
print counts
'''
'''
def get_counts2(se):
    counts = defaultdict(int) #创建了一个初始化的字典
    for x in se:
        counts[x] = counts[x] + 1
    return counts
counts = get_counts2(time_zone)
print counts
print counts['America/New_York']
print len(time_zone)
'''
'''
def top_counts(count_dict,n):
    value_key_pairs = [(count,tz) for tz,count in  count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[:-n:-1] #list倒序输出
print top_counts(counts,10)
'''
'''
counts  = Counter(time_zone)
print counts.most_common(10)
'''

frame = pd.DataFrame(records)
'''
print frame['a']
print len(frame['a'])
print frame['tz'][:10]
tz_counts = frame['tz'].value_counts()
print tz_counts[:10]
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknow'
tz_counts = clean_tz.value_counts()
print tz_counts[:10]
tz_counts[:10].plot(kind='barh',rot=0)
plt.show() # pycharm输入图像需要引用matplotlib.pyplot
print frame['a'][51]
'''
results = pd.Series([x.split()[0] for x in frame.a.dropna()])
# dropna与split函数的使用，Sreies函数的使用，frame.a.dropna的写法
# 对于一个 Series，dropna 返回一个仅含非空数据和索引值的 Series
# Series本质就是dict，而DataFrame可以看作多个Series的组合
# x.split()[0]在这里的用法:按空格划分，取出第一个元素即浏览器数据如Mozilla/5.0
'''
print frame.a.dropna()
print results[:5]
print results.value_counts()[:8]
#value_counts函数：进行计数
'''
cframe = frame[frame.a.notnull()]
# 去除缺失数据
# print cframe
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
# print operating_system
# np.where(condition,x,y)函数的用法：当满足条件时输出为x,否则为y,无后两个参数时输出为True
# cfream['a'].str.contains()函数的用法：contains函数起判断作用
# print type(cframe['a'])可以看出来此时数据类型是Series的，需要转化为str,否则报错'Series' object has no attribute 'contains'
# print type(operating_system)
'''
计算Windows系统的数量：
def counting(ex):
    counts = 0
    for x in ex:
        if x == 'Windows':
            counts+=1
    return counts
aa = counting(operating_system)
print aa
'''
by_tz_os = cframe.groupby(['tz', operating_system])
# groupby函数的使用:将'tz'数据按operating_system的数据进行分类
# print by_tz_os
agg_counts = by_tz_os.size().unstack().fillna(0)
# print type(agg_counts)
# agg_counts类型为DataFrame
# 将tz按照operating_system进行分组并计数并用unstack进行展开并填充na为0
# unstack()作用：展开为两列
# print type(by_tz_os)
# print agg_counts[:10]
'''
测试：
a = ['b','2','3','4']
series = pds.Series(['b',2,3,4])
ty_ze = series.groupby([series,a])
print type(ty_ze)
agg = ty_ze.size().unstack().fillna(0)
print agg
'''
indexer = agg_counts.sum(1).argsort()
# print indexer
# sum函数为默认axis = 0，是普通加和，axis = 1是按行加和,argsort是从小到大排序并返回下表
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
# 所以输出的应该为各tz的排名
count_subset = agg_counts.take(indexer)[-10:]
# 取出时区最多的值
# take(a,b)函数：取出a中符合b条件的值
# print count_subset
# count_subset.plot(kind = 'barh',stacked=True)
# plt.show()
# stacked用于产生堆积条形图
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plt.show()
# 此过程为将总体规范化为1，再显示各系统用户比例（比例展示）

