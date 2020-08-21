
# coding: utf-8

# In[1]:


import csv
import requests
import sys
import pandas as pd
import time
import datetime


# In[5]:


print(time.ctime(1596874200000))
print(time.ctime(1596874500000))


# In[2]:


# 正式環境 Prometheus url:
url = 'http://88.8.196.120:9090'

# UT 環境 Prometheus url:
# url = 'http://88.8.196.130:9090'


# prometheus query指令操作詳情請看以下這份文件
# https://prometheus.io/docs/prometheus/latest/querying/api/#instant-queries


# In[3]:


def GetMetrixNames(url):
    response = requests.get('{0}/api/v1/label/__name__/values'.format(url))
    names = response.json()['data']

    #Return metrix names
    return names


# In[4]:


# 設定欲查詢的metric需含哪些關鍵字
target1 = 'cpu'
target2 = 'memory'
target3 = 'thread'
target4 = 'response'
target_list = [target1,target2,target3,target4]
target = ['jvm_buffer_memory_used_bytes','jvm_gc_memory_allocated_bytes_total','jvm_memory_used_bytes', 'process_cpu_usage', 'system_cpu_count', 'system_cpu_usage']


# In[5]:


metrixNames=GetMetrixNames(url)
print('Metrix個數:',len(metrixNames))
print('-----------------------------')
cnt = 0
for i in metrixNames:
#     if i in target:
    print(i)
    cnt+=1

print(cnt)


# In[6]:


#example
list1 = []
count = 0
for i in metrixNames:
    print(i)
    for j in target_list:     
        if j in i:
            count += 1
            list1.append(i)


# In[7]:


for i in list1:
    print(i)
print(count)


# In[12]:


# 設定起始、終止時間及每幾秒查詢一次的時間
# step以秒回單位進去爬資料時，會被反爬蟲掉，因此以8小時為單位一天爬三次，step以 15s 來爬

start = '2020-07-12T16:00:00.000Z'
end = '2020-07-12T23:59:45.000Z'
step = '15s'

start2 = '2020-07-13T00:00:00.000Z'
end2 = '2020-07-13T07:59:45.000Z'
step2 = '15s'

start3 = '2020-07-13T08:00:00.000Z'
end3 = '2020-07-13T15:59:45.000Z'
step3 = '15s'


# In[13]:


# 服務斷掉及重啟: no data
# 07/07 00:00:00 - 08:00:00 


# In[14]:


# response = requests.get(url+'/api/v1/query_range?query=system_cpu_usage&start='+start+'&end=2020-07-27T04:05:00.781Z'+'&step='+step)
# print(response.json()['status'])


# In[ ]:


# setting date
day_data_name = '0713(1).csv'

# create csv
with open(day_data_name, 'w', newline='') as csvfile:
    label_list = ['name','timestamp_and_value']
    max_len_result = 0
    writer = csv.writer(csvfile)
    writeHeader=True
    for metrixResult in metrixNames:
#         if metrixResult in target:
        print(metrixResult)
#         if target1 in metrixResult or target2 in metrixResult:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start+'&end='+end+'&step='+step) #,params={'query': metrixResult})
        print(response)
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Build a list of all labelnames used.
            # Gets all keys and discard __name__
            labelnames = set()
            for result in results:
                labelnames.update(result['metric'].keys())

            # Canonicalize
            labelnames.discard('__name__')
            labelnames = sorted(labelnames)
            print(labelnames)        

            for i in labelnames:
                if i not in label_list:
                    label_list.append(i)

            for result in results:
                if len(result['values']) > max_len_result:
                    max_len_result = len(result['values'])
        
#         time.sleep(5)    

    for metrixResult in metrixNames:
#         if metrixResult in target:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start+'&end='+end+'&step='+step) #,params={'query': metrixResult})
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Write the header
            if writeHeader:
                writer.writerow(label_list)
                writeHeader = False

            # Write data into csv
            for result in results:
                cnt = 0
                l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]\
                if len(result['values']) < max_len_result:
                    l += ['NaN' for _ in range(max_len_result-len(result['values']))]
                for label in label_list:
                    l.append(result['metric'].get(label, 'NaN'))
                writer.writerow(l)                
#         time.sleep(5)

    print(max_len_result)
    csvfile.close()


# In[ ]:


# setting date
day_data_name2 = '0713(2).csv'


# create csv
with open(day_data_name2, 'w', newline='') as csvfile:
    label_list = ['name','timestamp_and_value']
    max_len_result = 0
    writer = csv.writer(csvfile)
    writeHeader=True
    for metrixResult in metrixNames:
#         if metrixResult in target:
        print(metrixResult)
#         if target1 in metrixResult or target2 in metrixResult:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start2+'&end='+end2+'&step='+step2) #,params={'query': metrixResult})
        print(response)
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Build a list of all labelnames used.
            # Gets all keys and discard __name__
            labelnames = set()
            for result in results:
                labelnames.update(result['metric'].keys())

            # Canonicalize
            labelnames.discard('__name__')
            labelnames = sorted(labelnames)
            print(labelnames)        

            for i in labelnames:
                if i not in label_list:
                    label_list.append(i)

            for result in results:
                if len(result['values']) > max_len_result:
                    max_len_result = len(result['values'])
        time.sleep(5)

    for metrixResult in metrixNames:
#         if metrixResult in target:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start2+'&end='+end2+'&step='+step2) #,params={'query': metrixResult})
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Write the header
            if writeHeader:
                writer.writerow(label_list)
                writeHeader = False

            # Write data into csv
            for result in results:
                cnt = 0
                l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]\
                if len(result['values']) < max_len_result:
                    l += ['NaN' for _ in range(max_len_result-len(result['values']))]
                for label in label_list:
                    l.append(result['metric'].get(label, 'NaN'))
                writer.writerow(l)                
        time.sleep(5)

    print(max_len_result)
    csvfile.close()


# In[ ]:


# setting date
day_data_name2 = '0713(3).csv'


# create csv
with open(day_data_name2, 'w', newline='') as csvfile:
    label_list = ['name','timestamp_and_value']
    max_len_result = 0
    writer = csv.writer(csvfile)
    writeHeader=True
    for metrixResult in metrixNames:
#         if metrixResult in target:
        print(metrixResult)
#         if target1 in metrixResult or target2 in metrixResult:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start3+'&end='+end3+'&step='+step3) #,params={'query': metrixResult})
        print(response)
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Build a list of all labelnames used.
            # Gets all keys and discard __name__
            labelnames = set()
            for result in results:
                labelnames.update(result['metric'].keys())

            # Canonicalize
            labelnames.discard('__name__')
            labelnames = sorted(labelnames)
            print(labelnames)        

            for i in labelnames:
                if i not in label_list:
                    label_list.append(i)

            for result in results:
                if len(result['values']) > max_len_result:
                    max_len_result = len(result['values'])
        time.sleep(5)

    for metrixResult in metrixNames:
#         if metrixResult in target:

        # now its hardcoded from time to time
        response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start2+'&end='+end2+'&step='+step2) #,params={'query': metrixResult})
        if response.json()['status'] ==  'success':
            results = response.json()['data']['result']

            # Write the header
            if writeHeader:
                writer.writerow(label_list)
                writeHeader = False

            # Write data into csv
            for result in results:
                cnt = 0
                l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]\
                if len(result['values']) < max_len_result:
                    l += ['NaN' for _ in range(max_len_result-len(result['values']))]
                for label in label_list:
                    l.append(result['metric'].get(label, 'NaN'))
                writer.writerow(l)                
        time.sleep(5)

    print(max_len_result)
    csvfile.close()


# In[ ]:


# release abundant memory
import gc
gc.collect()


# In[ ]:


# with open('up.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for metrixResult in metrixNames:
#         if metrixResult == 'up':

#             # now its hardcoded from time to time
#             response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start+'&end='+end+'&step='+step) #,params={'query': metrixResult})
#             results = response.json()['data']['result']

#             # Build a list of all labelnames used.
#             # Gets all keys and discard __name__
#             labelnames = set()
#             for result in results:
#                 labelnames.update(result['metric'].keys())

#             # Canonicalize
#             labelnames.discard('__name__')
#             labelnames = sorted(labelnames)
#             print(labelnames)

#             # Write the header
#             writer.writerow(['name', 'timestamp_and_value'] + labelnames)

#             # Write data into csv
#             for result in results:
#                 l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]                
#                 for label in labelnames:
#                     l.append(result['metric'].get(label, 'NaN'))
#                 writer.writerow(l)

#     csvfile.close()


# In[ ]:


# with open('time_interval_missingHeader.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writeHeader=True
#     for metrixResult in metrixNames:
#         if target1 in metrixResult or target2 in metrixResult:

#             # now its hardcoded from time to time
#             response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start+'&end='+end+'&step='+step) #,params={'query': metrixResult})
#             results = response.json()['data']['result']

#             # Build a list of all labelnames used.
#             # Gets all keys and discard __name__
#             labelnames = set()
#             for result in results:
#                 labelnames.update(result['metric'].keys())
            
#             # Canonicalize
#             labelnames.discard('__name__')
#             labelnames = sorted(labelnames)
#             print(labelnames)
            
#             # Write the header
#             if writeHeader:
#                 writer.writerow(['name', 'timestamp_and_value'] + labelnames)
#                 writeHeader=False

#             # Write data into csv
#             for result in results:
#                 l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]                
#                 for label in labelnames:
#                     l.append(result['metric'].get(label, 'NaN'))
#                 writer.writerow(l)

#     csvfile.close()


# In[ ]:


# with open('time_interval_withHeader.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for metrixResult in metrixNames:
#         if target1 in metrixResult or target2 in metrixResult:

#             # now its hardcoded from time to time
#             response = requests.get(url+'/api/v1/query_range?query='+metrixResult+'&start='+start+'&end='+end+'&step='+step) #,params={'query': metrixResult})
#             results = response.json()['data']['result']

#             # Build a list of all labelnames used.
#             # Gets all keys and discard __name__
#             labelnames = set()
#             for result in results:
#                 labelnames.update(result['metric'].keys())
            
#             # Canonicalize
#             labelnames.discard('__name__')
#             labelnames = sorted(labelnames)
#             print(labelnames)        
            
            
#             # Write the header
#             writer.writerow(['name', 'timestamp_and_value'] + labelnames)
                
#             # Write data into csv
#             for result in results:
#                 l = [result['metric'].get('__name__', '')] + result['values']  #values = [timestamp + value]                
#                 for label in labelnames:
#                     l.append(result['metric'].get(label, 'NaN'))
#                 writer.writerow(l)

#     csvfile.close()

