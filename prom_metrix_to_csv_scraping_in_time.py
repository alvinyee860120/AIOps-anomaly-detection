
# coding: utf-8

# In[1]:


import csv
import requests
import sys
import pandas as pd
import time
import datetime


# In[2]:


# 正式環境 Prometheus url:
url = 'http://88.8.196.120:9090'

# UT 環境 Prometheus url:
# url = 'http://88.8.196.130:9090'

# 指令操作詳情請看以下這份文件
# https://prometheus.io/docs/prometheus/latest/querying/api/#instant-queries


# In[3]:


def GetMetrixNames(url):
    response = requests.get('{0}/api/v1/label/__name__/values'.format(url))
    names = response.json()['data']
    #Return metrix names
    return names


# In[4]:


metrixNames=GetMetrixNames(url)
print('Metrix個數:',len(metrixNames))
print('-----------------------------')
for i in metrixNames:
    print(i)


# In[6]:


with open('past_time_allHeader.csv', 'w', newline='') as csvfile:
    label_list = ['name','timestamp_and_value']
    max_len_result = 0
    writeHeader = True
    writer = csv.writer(csvfile)
    for metrixResult in metrixNames:

        # request 從當下這個時間點往前推多久時間的 data
        response = requests.get('{0}/api/v1/query'.format(url),params={'query': metrixResult+'[1h]'})
        print(response)
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

        # Build a list of all labelnames used.
        # Gets all keys and discard __name__
        labelnames = set()
        for result in results:
            labelnames.update(result['metric'].keys())

        # Canonicalize
        labelnames.discard('__name__') #由於已有命名name，故捨棄重複命名項
        labelnames = sorted(labelnames)
        print(labelnames)

        for i in labelnames:
            if i not in label_list:
                label_list.append(i)
        for result in results:
            if len(result['values']) > max_len_result:
                max_len_result = len(result['values'])
            
    for metrixResult in metrixNames:

        # now its hardcoded from time to time
        response = requests.get('{0}/api/v1/query'.format(url),params={'query': metrixResult+'[1h]'})
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

        # Write the header
        if writeHeader:
            writer.writerow(label_list)
            writeHeader = False

        # Write data into csv
        for result in results:
            l = [result['metric'].get('__name__', '')] + result['values']
            if len(result['values']) < max_len_result:
                l += ['NaN' for _ in range(max_len_result-len(result['values']))]
            for label in label_list:
                l.append(result['metric'].get(label, 'NaN'))
            writer.writerow(l)

    csvfile.close()

