
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/30 16:27
# @Author  : sch
# @File    : test7.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import json

# 合并数据集
# 数据库风格的DataFrame合并
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
df1
df2
pd.merge(df1, df2)
pd.merge(df1, df2, on='key')

df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
df3
df4
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# 注意merge默认删除index中非交部分
pd.merge(df1, df2, how='outer')
# 保留并集

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
df1
df2
pd.merge(df1, df2, on='key', how='left')
# 多对多连接产生的是行的笛卡尔积

# 要根据多个键进行合并，传入一个由列名组成的列表即可
left =DataFrame({'key1': ['foo', 'foo', 'bar'],
                 'key2': ['one', 'two', 'one'],
                 'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
left
right
pd.merge(left, right, on=['key1', 'key2'], how='outer')

# 对重复列名的处理
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

# 索引上的合并——DataFrame的链接键位于索引中
left1 =DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                  'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
left1
right1
pd.merge(left1, right1, left_on='key', right_index=True)
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

# 层次化索引情况
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
lefth
righth
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                  columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13., 14.]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
left2
right2
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)
left2.join(right2, how='outer')
left1.join(right1, on='key')

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
left2.join([right2, another])
left2.join([right2, another], how='outer')

# 轴向连接
arr = np.arange(12.).reshape((3, 4))
arr
np.concatenate([arr, arr], axis=1)

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3])
pd.concat([s1, s2, s3], axis=1)
s4 =pd.concat([s1 * 5, s3])
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])
result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
result.unstack()
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])
df1
df2
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
pd.concat({'level1': df1, 'level2': df2}, axis=1)
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])

# 与当前分析工作无关的DataFrame索引
df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
df1
df2
pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df2])

# 合并重叠数据
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
a
b
np.where(pd.isnull(a), b, a)
# where函数的用法
b[:-2].combine_first(a[2:])
# combine_first:在缺失的地方打补丁
df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2)

# 重塑和轴向旋转
# 重塑层次化索引
data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data
result = data.stack()
result
result.unstack()
result.unstack(0)
result.unstack('state')

s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)

df = DataFrame({'left': result, 'right': result + 5},
               columns=pd.Index(['left', 'right'], name='side'))
df
df.unstack('state')
df.unstack('state').stack('side')

# 将“长格式”旋转为“宽格式”
'''
pivoted = ladta.pivot('data', 'item', 'value')
pivoted.head()
ldata['value2'] = np.random.randn(len(ldata))
ladata[:10]
ladata.pivot('data', 'item')
pivoted[:5]
pivoted['value'][:5]
unstacked = ladata.set_index(['data, 'item']).unstack('item')
'''

# 数据转换
# 移除重复数据
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data
data.duplicated()
data.drop_duplicates()

data['v1'] = range(7)
data.drop_duplicates(['k1'])
# data.drop_duplicates(['k1', 'k2'], take_last=True)

# 利用函数或映射进行数据转换
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                           'Pastrami', 'corned beef', 'Bacon',
                           'pastrami', 'honey ham', 'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {'bacon': 'pig', 'pulled pork': 'pig',
                 'pastrami': 'cow', 'corned beef': 'cow',
                 'honey ham': 'pig', 'nova lox': 'salmon'}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data
data['food'].map(lambda x: meat_to_animal[x.lower()])

# 替换值
data = Series([1., -999., 2., -999., -1000., 3.])
data
data.replace(-999, np.nan)
data.replace([-999, -1000], np.nan)
data.replace([-999, -1000], [np.nan, 0])
data.replace({-999: np.nan, -1000: 0})

# 重命名轴索引
data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.index.map(str.upper)
data.index = data.index.map(str.upper)
data
data.rename(columns=str.title, index=str.upper)
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data

# 离散化和面元划分
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats
cats.codes
pd.value_counts(cats)
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
data = np.random.rand(20)
pd.cut(data, 4, precision=2)
# precision设置的是保留位数
data = np.random.randn(1000)
cats = pd.qcut(data, 4)
cats
cats.value_counts()
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]).value_counts()

# 检测和过滤异常值
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data
data.describe()
col = data[3]
col[np.abs(col) > 3]
data[(np.abs(data) > 3).any(axis=1)]
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
# np.sign:符号函数

# 排列和随机采样
df = DataFrame(np.arange(5 * 4).reshape(5, 4))
df
sampler = np.random.permutation(5)
sampler
df
df.take(sampler)
df.take(np.random.permutation(len(df)))[:3]
bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)
draws = bag.take(sampler)
draws

# 计算指标/哑变量
# 另一种常用于统计建模或机器学习的转换方式是，将分类变量转换为
# ‘哑变量矩阵’或‘指标矩阵’。如果DataFrame的某一列中含有k个
# 不同的值，则可以派生出一个k列矩阵或DataFrame
# pandas有一个get_dummies函数可以进行实现
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy
df
# df['data1']的type为Series，df[['data1']]的type为DataFrame

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('test/test2_2/movies.dat', sep='::',
                       header=None, names=mnames, engine='python')
movies[:10]
#要为genre添加指标变量的时候需要先进性数据规整。
#首先把所有genres提取出来
genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))
dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
for i, gen in enumerate(movies.genres):
    dummies.ix[i, gen.split('|')] = 1
# enumerate函数：将可循环序列sequence以start开始分别列出序列数据和数据下标
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]

values = np.random.rand(10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))
DataFrame(values).join(pd.get_dummies(pd.cut(values, bins)))

# 字符串操作
# 字符串对象方法
val = 'a,b, guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
# strip函数修剪空白字符
pieces
first, second, third = pieces
first + '::' + second + '::' + third
'::'.join(pieces)
'guido' in val
val.index(',')
val.find(':')
val.count(',')
val.replace(',', '::')
val.replace(',', '')

# 正则表达式 regex
# re模块的函数可以分为三个大类：模式匹配、替换以及拆分。一个regex描述了需要在文本中定为的一个
# 模式，它可以用于许多目的。
# 我们先来看一个简单的例子：假设我想要拆分一个字符串，分隔符为数量不定的一组空白符（制表符、空
# 格、换行符等）。描述一个或多个空白符的regex是/s+：
import re
text = 'foo   bar\t baz  \tqux'
re.split('\s+', text)
regex = re.compile('\s+')
regex.split(text)
regex.findall(text)

text = '''Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
'''
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)
# re.IGNORECASE使得对大小写不敏感
regex.findall(text)
m = regex.search(text)
m
text[m.start():m.end()]
print(regex.sub('REDACTED', text))

pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
m = regex.match('wesm@bright.net')
m.groups()
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))


# USDA食品数据库
db = json.load(open('test/test7/foods-2011-10-03.json'))
len(db)
db[0].keys()
db[0]['nutrients'][0]
nutrients = DataFrame(db[0]['nutrients'])
nutrients[:7]

info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)
info[:5]
info
info.group.value_counts()

nutrients = []
for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
nutrients
nutrients.duplicated().sum()
nutrients = nutrients.drop_duplicates()
col_mapping = {'description': 'food',
               'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info

ndata = pd.merge(nutrients, info, on='id', how='outer')
ndata
ndata.ix[30000]

result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)