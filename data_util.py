#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:19:10 2018

@author: wangyikai
"""
import pandas as pd
import numpy as np
data = pd.read_excel('data.xlsx',sheetname=None)['工作表1']
j = 0
k = 0
for i in range(len(data)):
    if data['端口'][i]=='PC趋势':
        j += 1
    else:
        k += 1
n = 1.04
days = list(data['日期'])
features = list(data['种类'])
numbers = list(data['数量'])
types = list(data['端口'])
all_features = list(set(features))
all_days = list(sorted(set(days)))[1:]
data_dict = {}
missed = 0
for i in range(len(all_features)):
    temp = []
    for j in range(len(all_days)):
        index_days = [m for m, x in enumerate(days) if x == all_days[j]]
        index_features = [m for m, x in enumerate(features) if x == all_features[i]]
        index_needed = []
        for k in index_days:
            if k in index_features:
                index_needed.append(k)
        if len(index_needed)==2:
            if types[index_needed[0]]=='PC趋势':
                if numbers[index_needed[0]]>10:
                    if numbers[index_needed[1]]>10:
                        number = numbers[index_needed[0]]+numbers[index_needed[1]]
                    else:
                        number = 1.96 * numbers[index_needed[0]]
                else:
                    if numbers[index_needed[1]]>10:
                        number = 2.04 * numbers[index_needed[1]]
                    else:
                        number = 0
                        missed += 1
                        print('No data for '+str(all_days[j])+' for '+str(all_features[i]))
            else:
                if numbers[index_needed[0]]>10:
                    if numbers[index_needed[1]]>10:
                        number = numbers[index_needed[0]]+numbers[index_needed[1]]
                    else:
                        number = 2.04 * numbers[index_needed[0]]
                else:
                    if numbers[index_needed[1]]>10:
                        number = 1.96 * numbers[index_needed[1]]
                    else:
                        number = 0
                        missed += 1
                        print('No data for '+str(all_days[j])+' for '+str(all_features[i]))
        elif len(index_needed)==1:
            if types[index_needed[0]]=='PC趋势':
                if numbers[index_needed[0]]>10:
                    number = 1.96 * numbers[index_needed[0]]
                else:
                    number = 0
                    missed += 1
                    print('No data for '+str(all_days[j])+' for '+str(all_features[i]))
            else:
                if numbers[index_needed[0]]>10:
                    number = 2.04 * numbers[index_needed[0]]
                else:
                    number = 0
                    missed += 1
                    print('No data for '+str(all_days[j])+' for '+str(all_features[i]))
        else:
            print('No data for '+str(all_days[j])+' for '+str(all_features[i]))
            number = 0
            missed += 1
        temp.append([all_days[j], number])
    data_dict[all_features[i]] = temp
changed_dict = {}
wwyykk = ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
          '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12']   
for feat in all_features:
    changed = []
    temp = data_dict[feat]
    for dd in wwyykk:
        temp1 = []
        for i in range(len(temp)):
            if dd in temp[i][0]:
                temp1.append(temp[i][1])
        if len(temp1)==30:
            changed.append(temp1)
        elif len(temp1)<30:
            if len(temp1)==28:
                temp1.append(0)
                temp1.append(0)
            else:
                temp1.append(0)
            changed.append(temp1)
        else:
            changed.append(temp1[:30])
    changed = np.array(changed)
    mean_temp = np.mean(changed[np.where(changed!=0)])
    changed[np.where(changed==0)] = mean_temp
    changed_dict[feat] = changed
                       
                
    