#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27 11:42
# @Author  : sch
# @File    : test5.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# Series
obj = Series([4, 7, -5, 3])
obj

obj.values
obj.index

obj2 = Series([4, 7 , -5, 3], index=['a', 'b', 'c', 'd'])
obj2
obj2.index

obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]

obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)

'b' in obj2
'e' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah':5000}
obj3 = Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

pd.isnull(obj4)
pd.notnull(obj4)
obj.isnull()

obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

# DataFrame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame
DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = DataFrame(data, columns=['year', 'state','pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2
frame2.columns
frame2['state']
frame2['year']
frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(5.)
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
frame2['estern'] = frame2.state == 'Ohio'
frame2
del frame2['estern']
frame2.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3
frame3.T
DataFrame(pop, index=[2001, 2002, 2003])
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3
frame3.values
frame2.values

# 索引对象
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index

frame3
'Ohio' in frame3.columns
'2003' in frame3.index

# 基本功能
# 重新索引_reindex
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj.index
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
obj2.fillna(0)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
# ffill:前向填充值
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill')
frame.ix[['a', 'b', 'c', 'd'], states]

# 丢弃指定轴上的项
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
obj.drop(['d', 'c'])
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.drop(['Colorado', 'Ohio'])
data.drop(['two', 'four'], axis=1)

# 索引、选取和过滤
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1]
obj[2:4]
obj['b': 'c']

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data[:2]
data[data['three'] > 5]
data < 5
data[ data< 5 ] = 0
data

# 从行上使用标签索引：ix/loc
data.loc['Colorado', ['two', 'three']]
data.ix[2]
data.ix[['Colorado', 'Utah'], [3, 0, 1]]
data.ix[data.three > 5, :3]

# 算术运算与数据对齐
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Colorado'])
df1 + df2

# 在算术方法中填充值
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1 + df2
df1.add(df2, fill_value=0)
df1.reindex(columns=df2.columns, fill_value=0)

# DataFrame和Series之间的运算
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]
frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame - series
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2
series3 = frame['d']
frame.sub(series3, axis=0)

# 函数应用和映射
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)
frame.apply(lambda x: x.max() - x.min())
frame.apply(lambda x: x.max() - x.min(), axis=1)
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)
format = lambda x: '%.2f' % x
frame.applymap(format)
frame['e'].map(format)
# map方法：对Series进行apply

# 排序和排名
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
frame = DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=1, ascending=False)

obj = Series([4, 7, -3, 2])
obj.sort_values()

obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
# 对某列进行排序
frame.sort_values(by='b')
frame.sort_values(by=['a', 'b'])

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method='first')
obj.rank(ascending=False, method='max')
frame = DataFrame({'b': [4.3, 7, -3, 2], 'a':[0, 1, 0, 1],
                   'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis=1)

# 带有重复值的轴索引
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
obj.index.is_unique
obj['a']
obj['c']
df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df.loc['b']

# 汇总和计算描述统计
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df
df.sum()
df.sum(axis=1)
df.mean(axis=1, skipna=False)
'''
约简方法： 
axis  所选轴，行为0，列为1
skipna 排除缺失值，默认为True
'''

df.idxmax()
# 间接统计的方法：返回达到最大或者最小值的索引idexmin与idxmax

df.cumsum()
# 累加

df.describe()

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
'''
描述和汇总统计的方法
count               非NA值的数量
describe            针对Series或各DataFrame列计算汇总统计
min、max            计算最小值和最大值
argmin、argmax      计算能够获取到最小值和最大值的索引位置（整数）
idxmin、idxmax      计算能够获取到最小值和最大值的索引值
qutantile           计算样本的分位数（0到1）
sum                 值的总和
mean                值的平均数
median              值的算术中位数（50%分位）
mad                 根据平均值计算平均绝对离差
var                 样本值的方差
std                 样本值的标准差
skew                样本值的偏度（三阶矩）
kurt                样本值的峰度（四阶矩）
cumsum              样本值的累计和
cummin、cummax      样本值的累计最大值和累计最小值
cumprod             样本值的累计积
diff                计算一阶差分（对时间序列很有用）
pct_change          计算百分数变化
'''
'''
相关系数与协方差
有些汇总统计（如相关系数和协方差）是通过参数对计算出来的。
获取的数据集格式为：
all_data
{'AAPL':                  Open       High        Low      Close  Adj Close     Volume
Date
2000-01-03   3.745536   4.017857   3.631696   3.997768   3.610740  133949200
2000-01-04   3.866071   3.950893   3.613839   3.660714   3.306317  128094400
链接不到yahoo，pandas版本更改

import pandas.io.data as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')
price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})
接下来计算价格的百分数变化
teturns = price.pct_change()
Series的corr方法用于计算两个Series中重叠的、非NA的、按索引对齐的值
的相关系数，与此类似，cov用于计算协方差
returns.MSFT.corr(returns.IBM)
DataFrame的corr和cov方法将以DataFrame的形式返回完整的相关系数或协方差矩阵
returns.corr()
returns.cov()
利用DataFrame的corrwith方法，你可以计算其行或列跟另一个Series或DataFrame
之间的相关系数。传入一个Series将会返回一个相关系数值Series（针对各列进行计算）
returns.corrwith(returns.IBM)
传入一个DataFrame则会计算按列名配对的相关系数。这里计算百分比变化与成交量
returns.corrwith(volume)
传入axis=1则可按行计算
'''

# 唯一值、值计数以及成员资格
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj
uniques = obj.unique()
uniques
# 得到Series中的唯一值数组

obj.value_counts()
# 用于计算一个Series中各值出现的频率
pd.value_counts(obj.values, sort=False)

mask = obj.isin(['b', 'c'])
mask
obj[mask]
# isin函数 判断成员资格

data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data
result = data.apply(pd.value_counts)
result
# 得到DataFrame中多个相关列的一张柱状图
# value_counts函数可以通过apply方法传给data

# 处理缺失数据
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
string_data.isnull()
string_data[0] = None
string_data.isnull()
# NA的处理方法：dropna、fillna

# 滤除缺失数据
from numpy import nan as NA

data = Series([1, NA, 3.5, NA, 7])
data.dropna()
data[data.notnull()]

data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                 [NA, NA, NA], [NA, 6.5, 3.]])
data
cleaned = data.dropna()
cleaned
# dropna将会删去所有存在nan的行
# 传入how='all'将只丢弃全为NA的行
data.dropna(how='all')
# 如果想丢弃列，只需要传入axis=1
data[4] = NA
data
data.dropna(axis=1, how='all')

# 另一个滤除DataFrame行的问题涉及时间序列数据，假设我们只想留下一部分
# 观测数据，可以用thresh参数实现此目的
df = DataFrame(np.random.randn(7, 3))
df
df.ix[:4, 1] = NA;df.ix[:2, 2] = NA
df
df.dropna(thresh=3)
# thresh参数的作用

# 填充缺失数据
df.fillna(0)
df.fillna({1: 0.5, 3: -1})
# 通过字典调用，对不同的列填充不同的值

df = DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA;df.iloc[4:, 2] = NA
df
df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)
# limit 可以连续填充的最大数量

data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())

# 层次化索引
# 层次化索引是pandas的一项重要功能，它使你能在一个轴上拥有多个索引级别
# 你可以以低维度形式处理高维度数据
data = Series(np.random.randn(10),
              index = [['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                       [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data
data.index
data['b']
data['b':'c']
data.loc[['b', 'd']]
data[:, 2]
data.unstack()
data.unstack().stack()

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'],[1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])
frame
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame
frame['Ohio']

# 重排分级顺序
frame.swaplevel('key1', 'key2')
frame.sortlevel(1)
frame.swaplevel(0, 1).sort_index()

# 根据级别汇总统计
frame.sum(level='key2')
frame.sum(level='color', axis=1)

# 使用DataFrame的列
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame
frame2 = frame.set_index(['c', 'd'])
frame2
frame.set_index(['c', 'd'], drop=False)
frame2.reset_index()
# set_index与reset_index作用相反

# 其他有关pandas的话题

# 整数索引
ser = Series(np.arange(3.))
ser
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]
ser.loc[:1]

# 面板数据 Panel
