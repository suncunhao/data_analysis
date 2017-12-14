#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/18 15:05
# @Author  : sch
# @File    : test2_2.py

import pandas as pd

#读取数据：
unames = ['user_id','gender','age','occupation','zip']
#这里要注意数据对齐，不然会出错
users = pd.read_table('users.dat',sep='::',header=None,names=unames,engine = 'python')
#观察数据结构，数据间以::分隔,header为用作列名的行号，默认为0（第一行），没有的话设置为None，此时names用来规定列名
#print users.head()
# #利用python的切片语法观察前几行数据的规律，是否划分正确
#为什么zip与下个user_id间没有空缺也能正确读取（txt打开）
#不加最后的engine = 'python'会报错，但是可以忽略
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('ratings.dat',sep='::',header=None,names=rnames,engine = 'python')
#类似上面
mnames = ['movie_id','title','genres']
movies = pd.read_table('movies.dat',sep='::',header=None,names=mnames,engine='python')
#三张表进行分析比较困难，考虑合并：先用pandas的merge函数将rating和users合并，再合并movies
#pandas会根据列名的重叠情况推断出哪些列是合并key
data = pd.merge(pd.merge(ratings,users),movies)
#print data
#print data.loc[0]
#输出特定行需要使用ix，输出特定列不需要只需要如print data['zip']
#也可以使用print data.loc[0]
mean_ratings = pd.pivot_table(data,index='title',values='rating',columns='gender',aggfunc='mean')
# mean_ratings2 = pd.pivot_table(data,index=['title','gender'],values='rating',aggfunc='mean')
# print mean_ratings.head()
# print mean_ratings2.head()
#pandas版本更新问题:rows改为index,col写为全称columns
#pivot_table函数：http://blog.csdn.net/youngbit007/article/details/53039390
#参数：数据来源data、总索引title、针对值rating、次要关注gender、算法选择mean，多个字段使用中括号[]
#index与columns的区别：一个以行排列，一个以列排列
'''
以下两个写法一样：
print pd.pivot_table(data,index=['movie_id','occupation'],values=['rating'])
print data.pivot_table('rating',index=['movie_id','occupation'])
'''
#删除评论数目低于250条的：
ratings_by_title = data.groupby('title').size()
#将title对data分组并计数：使用groupby函数对title合并再用size()求和
#print ratings_by_title
active_titles = ratings_by_title.index[ratings_by_title >= 250]
#index函数：按索引进行查找
#print active_titles
mean_ratings = mean_ratings.ix[active_titles]
#print mean_ratings
top_female_ratings = mean_ratings.sort_values(by='F',ascending=False)
#pandas在此版本的sort_index函数没有by参数
#将sort_index函数改为sort_value函数，没有报错
#sort_index函数：对行、列的索引进行排序，ascending为排序的方式，T为升序F为降序
#print top_female_ratings[:10]

#接下来计算评分分歧
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
#直接加入一个diff列，用于存放评分差
sorted_by_diff = mean_ratings.sort_values(by='diff')
#sorted_by_diff2 = mean_ratings.sort_index(by='diff',ascending=False)
#这里可以使用ascending=False进行降序
#print sorted_by_diff[:15]
##print sorted_by_diff2[:15]输出结果与上面的sort_by_diff相同
rating_std_by_title = data.groupby('title')['rating'].std()
#std:计算标准差，这里注意()[]的用法
rating_std_by_title = rating_std_by_title.ix[active_titles]
#筛选有效评分
#print rating_std_by_title
# print type(rating_std_by_title)
print(rating_std_by_title.sort_values(ascending=False)[:10])
#原书中为order，由于pandas版本更新，请使用sort_values函数


