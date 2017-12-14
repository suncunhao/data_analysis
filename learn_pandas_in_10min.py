#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/23 9:22
# @Author  : sch
# @File    : learn_pandas_in_10min.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 一、创建对象
# 1.可以通过传递一个list对象来创建一个Series，pandas会默认创建整形索引
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# 2.通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.rand(6, 4), index=dates, columns=['A','B','C','D'])

# 3.通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame
df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D':np.array([3] * 4, dtype='int32'),
                    'E':pd.Categorical(['test', 'train', 'test', 'train']),
                    'F':'foo'})

# 4.查看不同列的数据类型
df2.dtypes

# 二、查看数据
# 1.查看frame中头和尾部的行
df.head()
df.tail(3)

# 2.显示索引、列和底层的numpy数据
df.index
df.columns
df.values

# 3.describe()函数对于数据的快速统计汇总
df.describe()

# 4.对数据的转置
df.T

# 5.按轴进行排序
df.sort_index(axis=1, ascending=False)

# 6.按值进行排序
df.sort_values(by='B')
# pandas版本问题，与教程中的不符

# 三、选择
# 获取
# 1.选择一个单独的列，这将会返回一个Series，等同于df.A
df['A']

# 2.通过[]进行选择，这将会对行进行切片
df[0:3]
df['20130102':'20130104']
# 第二个切片包含了首位

# 通过标签选择
# 1.使用标签来获取一个交叉的区域
df.loc[dates[0]]

# 2.通过标签来在多个轴上进行选择
df.loc[:, ['A', 'B']]

# 3.标签切片
df.loc['20130102':'20130104', ['A', 'B']]

# 4.对于返回的对象进行维度缩减
df.loc['20130102', ['A', 'B']]

# 5.获取一个标量
df.loc[dates[0], 'A']

# 6.快速访问一个标量
df.at[dates[0], 'A']

# 通过位置选择
# 1.通过传递数值进行位置选择（选择的是行）
df.iloc[3]

# 2.通过位置进行切片，与numpy/python中的情况类似
df.iloc[3:5, 0:2]

# 3.通过指定一个位置的列表，与numpy/python中的情况类似
df.iloc[[1, 2, 4], [0, 1]]

# 4.对行进行切片
df.iloc[1:3, :]

# 5.对列进行切片
df.iloc[:, 1:3]

# 6.获取特定的值
df.iloc[1, 1]
df.iat[1, 1]

# 布尔索引
# 1.使用一个单独列的值来选择数据
df[df.A > 0.5]

# 2.使用where操作来选择数据
df[df > 0.5]

# 3.使用isin()方法来过滤
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2
df2[df2['E'].isin(['two', 'four'])]

# 设置
# 1.设置一个新的列
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
s1
df['F'] = s1

# 2.通过标签设置新的值
df.at[dates[0], 'A']

# 3.通过位置设置新的值
df.iat[0, 1] = 0

# 4.通过一个numpy数组设置一组新值
df.loc[:, 'D'] = np.array([5] * len(df))

# 5.通过where操作来设置新的值
df2 = df.copy()
df2[df2 > 0] = -df2
df2

# 四、缺失值处理
# 在pandas中，使用np.nan来代替缺失值，这些值将默认不会包含在计算中
# 1.reindex()方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝：
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

# 2.去掉包含缺失值的行
df1.dropna(how='any')

# 3.对缺失值进行填充
df1.fillna(value=5)

# 4.对缺失值进行布尔填充
pd.isnull(df1)

# 五、相关操作
# 统计（相关操作通常情况下不包括缺失值）
# 1.执行描述性统计
df.mean()

# 2.在其他轴上进行相同的操作
df.mean(1)
# 即对行进行统计，参数同axis=1

# 3.对于拥有不同维度，需要对其的对象进行操作。pandas会自动沿着指定的维度进行广播
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
# shift函数：向下平移2行，没有值的用nan代替
df.sub(s, axis='index')
# sub函数：对不同维度的pandas对象进行减法操作，减nan的会变成nan

# apply
# 1.对数据应用函数：
df.apply(np.cumsum)
# cumsum函数：累加

# 直方图
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

# 字符串方法
# Series对象在其str属性中配备了一组字符串处理方法，可以很容易的应用到数组中的各个元素
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

# 六、合并
# pandas提供了大量的方法能够轻松地对Series，DataFrame和Panel对象进行各种符合各种逻辑关系的合并操作
# concat
df = pd.DataFrame(np.random.rand(10,4))
df
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# join类似于SQL类型的合并
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1,2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4,5]})
left
right
pd.merge(left, right, on='key')

# Append将一行连接到一个DataFrame上
df = pd.DataFrame(np.random.rand(8,4), columns=['A', 'B', 'C', 'D'])
df
s = df.iloc[3]
df.append(s, ignore_index=True)

# 七、分组
'''
对于 groupby 操作，我们通常是指以下一个或多个操作步骤：
splitting 按照一些规则将数据分为不同的组
applying 对于每组数据分别执行一个函数
combining 将结果组合到一个数据结构中
'''
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C':np.random.rand(8),
                   'D':np.random.rand(8)})
df

# 1.分组并对每个分组执行sum函数
df.groupby('A').sum()

# 2.通过多个列进行分组形成一个层次索引，然后执行函数
df.groupby(['A', 'B']).sum()

# 八、Reshaping
# stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.rand(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2

stacked = df2.stack()
# 扁平化
stacked.unstack()

stacked.unstack(1)
stacked.unstack(0)
#运行可以看出对不同的轴进行操作

# 数据透视表pivot_tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B':['A', 'B', 'C'] * 4,
                   'C':['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D':np.random.rand(12),
                   'E':np.random.rand(12)})
df
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
# 数据透视表的参数设置

# 九、时间序列
# pandas在对频率转换进行重新采样时具有简单、强大且高效的功能（如将按秒采样的数据转换为按5分钟为单位进行采样的数据）
rng = pd.date_range('1/1/2012', periods=100, freq='S')
# 创造时间序列
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
# 创造Series
ts.resample('5Min').sum()
# pandas版本更新how in .resample() is deprecated, the new syntax is .resample(...).sum()

# 1.时区表示
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')

# 2.时区转换
ts_utc.tz_convert('US/Eastern')

# 3.时间跨度转换
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
# 输出形式为月末

ps = ts.to_period()
ps
#输出形式为月份

ps.to_timestamp()
# 输出形式为月初

# 4.时期和时间戳之间的转换使得可以使用一些方便的算术函数
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()
# asfreq函数

# 十、categorical
# pandas在DataFrame中支持Categorical类型的数据
df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})

# 1.将原始的grade转换为Categorical数据类型
df['grade'] = df['raw_grade'].astype('category')

# 2.将Catego类型数据重命名为更有意义的名称
df['grade'].cat.categories = ['very good', 'good', 'very bad']

# 3.对类别进行重新排序，增加缺失的类别
df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium', 'good', 'very good'])
df['grade']

# 4.排序是按照categorical的顺序进行的而不是按照字典顺序进行
df.sort_values('grade')

# 5.对catego列进行排序时存在空的类别
df.groupby('grade').size()

# 十一、画图
ts = pd.Series(np.random.rand(1000), index=pd.date_range('1/1/2000', periods=1000))
ts_c = ts.cumsum()
ts_c.plot()

# 对于DataFrame来说，plot是一种将所有列及其标签进行绘制的简便方法
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df_c = df.cumsum()
plt.figure();df_c.plot();plt.legend(loc='best')
# plt函数的使用方法

# 十二、数据IO
# csv
# 1.写入csv文件
df.to_csv('foo.csv')

# 2.从csv文件中读取
pd.read_csv('foo.csv')

# HDF5
# 1.写入HDF5存储
df.to_hdf('foo.h5', 'df')

# 2.从HDF5存储中读取
pd.read_hdf('foo.h5', 'df')

# excel
# 1.写入excel文件
df.to_excel('foo.xlsx', sheet_name='Sheet1')

# 2.从excel文件中读取
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
