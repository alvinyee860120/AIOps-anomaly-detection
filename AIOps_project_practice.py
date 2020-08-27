
# coding: utf-8

# In[1]:


import keras 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten,Reshape
from keras.optimizers import Adam

import flask


# In[2]:


file = 'prometheus_standardized_data.csv'
df = pd.read_csv(file, parse_dates=['timestamp'],low_memory=False)


# In[3]:


df


# In[4]:


df.info()


# In[5]:


api1 = df.loc[df['job'] == 'mip-push-notify-counter'].values
api2 = df.loc[df['job'] == 'mip-svc-push-notify'].values
count = 0
for i in api1: 
    if i[0] == "process_cpu_usage": 
        count += 1
        
print(count)

count2 = 0
for i in api2:
    if i[0] == "process_cpu_usage": 
        count2 += 1
print(count2)


# In[6]:


# extract feature
job_name = df['job'].values
name = df['name'].values
id_ = df['id'].values

# setting target_api name
target_api = 'mip-push-notify-counter'
target_api2 = 'mip-svc-push-notify'


# In[7]:


def extract_target_index(api_name):
    index = []
    for i,v in enumerate(job_name):
        if v == api_name:
            index.append(i)
    print(len(index))
    return index


# In[8]:


# extract index from target api
index = extract_target_index(target_api)
index2 = extract_target_index(target_api2)


# In[9]:


def generate_list(index,value_list):
    dic = []
    for i in index:
        if value_list[i] not in dic:
            dic.append(value_list[i])
    print(dic)
    return dic


# In[10]:


# extract unique name & id from target api
name_dict = generate_list(index,name)
id_dict = generate_list(index,id_)

name_dict2 = generate_list(index2,name)
id_dict2 = generate_list(index2,id_)


# In[11]:


def helper(x):
    metric,time,value,id_ = [],[],[],[]
    for i in x:
        metric.append(i[0])
        time.append(i[1])
        value.append(i[2])
        id_.append(i[-1])
    return metric,time,value,id_


# In[12]:


metric1,time1,value1,id1 = helper(api1)
metric2,time2,value2,id2 = helper(api2)


# In[13]:


print(len(metric1),len(metric2))


# In[14]:


def haddle_missing_value(metric1,metric2,time1,time2,value1,value2,id1,id2):
    t_max = max(len(time1),len(time2))
    new_metric, new_timestamp, new_value,new_id = [],[],[],[]
    for i in range(t_max):
        if time1[i] != time2[i]:
            print('before:',time1[i],time2[i])
            time1.insert(i,time2[i])
            metric1.insert(i,metric2[i])
            value1.insert(i,0)
            id1.insert(i,id2[i])
            print('after:',time1[i],time2[i])
    
    for i in range(t_max):
        new_metric.append(metric1[i])
        new_timestamp.append(time1[i])
        new_value.append(value1[i])
        new_id.append(id1[i])
        
    print(len(new_metric),len(time2))
    print(len(new_timestamp),len(time2))
    print(len(new_value),len(value2))
    print(len(new_id),len(id2))
    return new_metric, new_timestamp, new_value, new_id


# In[15]:


# setting taregt_id & metric name for cpu
mem_target_id = ['PS Eden Space', 'PS Old Gen']
cpu_name = ['process_cpu_usage', 'system_cpu_count', 'system_cpu_usage']


# catch timestamp & corresponding value
def catch_target_timestamp(time,metric_name,metric,id_,t_id):
    listname = []
    cnt = 0
    for i,v in enumerate(time):
        if metric[i] == metric_name and id_[i] == t_id:
            listname.append(cnt)
            cnt += 1
    return listname

def catch_target_value(value,metric_name,metric,id_,t_id):
    listname = []
    for i,v in enumerate(value):
        if metric[i] == metric_name and id_[i] == t_id:
            listname.append(value[i])
    return listname

def catch_cpu_timestamp(time,metric_name,metric):
    listname = []
    cnt = 0
    for i,v in enumerate(time):
        if metric[i] == metric_name:
            listname.append(cnt)
            cnt += 1
    return listname

def catch_cpu_value(value,metric_name,metric):
    listname = []
    for i,v in enumerate(value):
        if metric[i] == metric_name:
            listname.append(value[i])
    return listname

def create_df(t,v):
    print(len(t),len(v))
    data = list(zip(t,v))
    tmp_df = pd.DataFrame(data,columns=['time','value'])
    return tmp_df

# plotting
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(x, y, color='tab:green')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
# 兩筆資料視覺化比對
def plot_compare(df,df2, x, y1, y2, title="", xlabel='Date', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(x, y1, label = target_api, color='tab:green')
    plt.plot(x, y2, label = target_api2, color='tab:orange')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.show()


# In[54]:


# 單筆資料設定threshold 作圖
#折線圖
def masked_plot(df,x,y,title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    threshold = 0.0065
    threshold2 = 0.0
    y = np.array(y)
    masked = np.ma.masked_less_equal(y, threshold)
    masked2 = np.ma.masked_less_equal(y, threshold2)
    plt.plot(y,color='k')
    plt.plot(masked2, color='tab:green', linewidth=2)
    plt.plot(masked, color= 'red', linewidth=2)
    
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.axhline(threshold, color='k', linestyle='--')
    plt.show()

# 點狀圖
def marker_plot(df,x,y,title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    # Plot the line
    threshold = 0.0065
    threshold2 = 0.0001
    y = np.array(y)
    plt.plot(x, y, color='tab:orange')
    # Add below threshold markers
    below_threshold = y < threshold
    below_threshold2 = y < threshold2
    plt.scatter(x[below_threshold], y[below_threshold], color='tab:green') 
    plt.scatter(x[below_threshold2], y[below_threshold2], color='k')
    
    # Add above threshold markers
    above_threshold = np.logical_not(below_threshold)
    plt.axhline(threshold, color='k', linestyle='--')
    plt.scatter(x[above_threshold], y[above_threshold], color='r')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[17]:


# 補齊 missing point
metric1, time1, value1, id1 = haddle_missing_value(metric1,metric2,time1,time2,value1,value2,id1,id2)


# In[18]:


for metric_name in name_dict:
    for i in id_dict:
        print(metric_name,'&',i)
        t = catch_target_timestamp(time1,metric_name,metric1,id1,i)
        v = catch_target_value(value1,metric_name,metric1,id1,i)
        tmp_df = create_df(t,v)

print('---------------------------------------------------------')

for metric_name in cpu_name:
    print(metric_name)
    t = catch_cpu_timestamp(time1,metric_name,metric1)
    v = catch_cpu_value(value1,metric_name,metric1)
    tmp_df = create_df(t,v)


# In[19]:


for metric_name in name_dict:
    for i in id_dict2:
        print(metric_name,'&',i)
        t = catch_target_timestamp(time2,metric_name,metric2,id2,i)
        v = catch_target_value(value2,metric_name,metric2,id2,i)
        tmp_df = create_df(t,v)

print('---------------------------------------------------------')

for metric_name in cpu_name:
    print(metric_name)
    t = catch_cpu_timestamp(time2,metric_name,metric2)
    v = catch_cpu_value(value2,metric_name,metric2)
    tmp_df = create_df(t,v)


# In[55]:


# 含異常值折線圖
metric_name = 'process_cpu_usage'
t = catch_cpu_timestamp(time1,metric_name,metric1)
v = catch_cpu_value(value1,metric_name,metric1)
if len(t) > 0:
    tmp_df = create_df(t,v)
    masked_plot(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  from time to time')


# In[56]:


# 含異常值 scatter plot
metric_name = 'process_cpu_usage'
t = catch_cpu_timestamp(time1,metric_name,metric1)
v = catch_cpu_value(value1,metric_name,metric1)
if len(t) > 0:
    tmp_df = create_df(t,v)
    marker_plot(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  from time to time')


# In[38]:


# 比較兩支API Memory usage
for metric_name in name_dict:
    for i in mem_target_id:
        t = catch_target_timestamp(time1,metric_name,metric1,id1,i)
        v = catch_target_value(value1,metric_name,metric1,id1,i)
        t_2 = catch_target_timestamp(time2,metric_name,metric2,id2,i)
        v_2 = catch_target_value(value2,metric_name,metric2,id2,i)
        if len(t) and len(t_2) > 0:
            tmp_df = create_df(t,v)
            tmp_df2 = create_df(t_2,v_2)
            plot_compare(tmp_df, tmp_df2, tmp_df.time, tmp_df.value, tmp_df2.value,
                    title= 'metric: '+metric_name+'  id:'+i+'  from time to time')


# In[39]:


# 比較兩支API cpu usage
for metric_name in cpu_name:
    t = catch_cpu_timestamp(time1,metric_name,metric1)
    v = catch_cpu_value(value1,metric_name,metric1)
    t_2 = catch_cpu_timestamp(time2,metric_name,metric2)
    v_2 = catch_cpu_value(value2,metric_name,metric2)
    if len(t) and len(t_2) > 0:
        tmp_df = create_df(t,v)
        tmp_df2 = create_df(t_2,v_2)
        plot_compare(tmp_df, tmp_df2, tmp_df.time, tmp_df.value, tmp_df2.value,
                title= 'metric: '+metric_name+'  from time to time')


# In[ ]:


# #'mip-push-notify-counter' Memory usage
# for metric_name in name_dict:
#     for i in mem_target_id:
#         t = catch_target_timestamp(time1,metric_name,metric1,id1,i)
#         v = catch_target_value(value1,metric_name,metric1,id1,i)
#         if len(t) > 0:
#             tmp_df = create_df(t,v)
#             plot_df(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  id:'+i+'  from time to time')


# In[ ]:


# # 'mip-push-notify-counter' cpu usage
# for metric_name in cpu_name:
#     t = catch_cpu_timestamp(time1,metric_name,metric1)
#     v = catch_cpu_value(value1,metric_name,metric1)
#     if len(t) > 0:
#         tmp_df = create_df(t,v)
#         plot_df(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  from time to time')


# In[ ]:


# # 'mip-svc-push-notify' Memory usage
# for metric_name in name_dict2:
#     for i in mem_target_id:
#         t = catch_target_timestamp(time2,metric_name,metric2,id2,i)
#         v = catch_target_value(value2,metric_name,metric2,id2,i)
#         if len(t) > 0:
#             tmp_df = create_df(t,v)
#             plot_df(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  id:'+i+'  from time to time')


# In[ ]:


# #'mip-svc-push-notify' cpu usage
# for metric_name in cpu_name:
#     t = catch_cpu_timestamp(time2,metric_name,metric2)
#     v = catch_cpu_value(value2,metric_name,metric2)
#     if len(t) > 0:
#         tmp_df = create_df(t,v)
#         plot_df(tmp_df, x=tmp_df.time, y=tmp_df.value, title= 'metric: '+metric_name+'  from time to time')


# In[ ]:


def create_dataset(name_dict, mem_target_id, cpu_name, time1, metric1, id1):
    value_collections = []
    column_name = []
    time_collections = []
    for metric_name in name_dict:
        for i in mem_target_id:
            t = catch_target_timestamp(time1,metric_name,metric1,id1,i)
            v = catch_target_value(value1,metric_name,metric1,id1,i)
            if len(t) and len(v) > 0:
                column_name.append(metric_name+'& '+i)
                value_collections.append(v)
                time_collections.append(time1[:len(t)])
        if metric_name in cpu_name:
            t2 = catch_cpu_timestamp(time1,metric_name,metric1)
            v2 = catch_cpu_value(value1,metric_name,metric1)
            column_name.append(metric_name)
            value_collections.append(v2)

    column_name.insert(0,'Time')
    data = list(zip(time_collections[0],value_collections[0],value_collections[1],value_collections[2]
                 ,value_collections[3],value_collections[4]))
    df = pd.DataFrame(data,columns=column_name)

    return df


# In[ ]:


new_df = create_dataset(name_dict, mem_target_id, cpu_name, time1, metric1, id1)
new_df2 = create_dataset(name_dict, mem_target_id, cpu_name, time2, metric2, id2)


# In[ ]:


new_df


# In[ ]:


new_df2


# In[ ]:


import gc
gc.collect()

