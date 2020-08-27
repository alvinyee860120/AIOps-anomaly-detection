
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import time


# In[33]:


file = 'time_interval_allHeader2.csv'
# file2 = 'past_time_allHeader.csv' 

def read_file(filename):
    data = []
    header = []
    with open( file,'r') as f:
        read = f.readlines()
        header = read[0][:-1].split(',')
        for i in read[1:]:
            data.append(i)
            
    return data, header


# In[34]:


data_body, data_head = read_file(file)
# data_body2, data_head2 = read_file(file2)


# In[35]:


print(data_head)
# print(data_head2)


# In[36]:


def create_data_list(datafile):
    data2list = []
    for i,v in enumerate(datafile):
        data2list.append(v.split(','))
        print(data2list[i][0])
    return data2list


# In[37]:


data_list = create_data_list(data_body)
# data_list2 = create_data_list(data_body2)

# print(data_list[0][1][2:])
# print(data_list[0][2][-2:])
print(len(data_list))
for i in data_list:
    print(len(i))


# In[38]:


def fill_subdata_in_dict(col,length):
    sub_list = [col for _ in range(length)]
    return sub_list


# In[39]:


# data_list[1]


# In[40]:


# a = fill_subdata_in_dict(data_list[0][0],241)
# print(len(a))


# In[41]:


def datalist_to_dictionry(data_list):
    name,timestamp,value,instance,job,id_ = {},{},{},{},{},{}
    for i,v in enumerate(data_list):
        sub_timestamp,sub_value = [], []
        for j,val in enumerate(v):
            if val[:2] == '"[':
                sub_timestamp.append(val[2:])
            elif val[-2:] == ']"':
                sub_value.append(val[2:-3])
        
        n = len(sub_timestamp)
        name[i] = fill_subdata_in_dict(v[0],n)
        timestamp[i] = sub_timestamp
        value[i] = sub_value
        id_[i] = fill_subdata_in_dict(v[-4],n)
        instance[i] = fill_subdata_in_dict(v[-3],n)
        job[i] = fill_subdata_in_dict(v[-2],n)
        
    return name, timestamp, value, instance, job, id_


# In[42]:


name, timestamp, value, instance, job, id_ = datalist_to_dictionry(data_list)
# name2, timestamp2, value2, instance2, job2, hostname2, application2, id_2, area2, pool2 = datalist_to_dictionry(data_list2)


# In[43]:


#check every column data has equal length
print(len(name[0]))
print(len(timestamp[0]))
print(len(value[0]))
print(len(instance[0]))
print(len(job[0]))
print(len(id_[0]))


# In[44]:


print(len(name))
print(len(timestamp))
print(len(value))
print(len(instance))
print(len(job))
print(len(id_))


# In[45]:


# change time(seconds) into local time
local = time.ctime(float(timestamp[0][0]))
print(local)


# In[46]:


def create_df_from_dict(name, timestamp, value, instance, job, id_):
    tmp_name,tmp_timestamp,tmp_value,tmp_instance,tmp_job,tmp_id = [],[],[],[],[],[]
    for i,v in name.items():
        for j,val in enumerate(v):
            print(val)
            tmp_name.append(val)
            tmp_timestamp.append(time.ctime(float(timestamp[i][j])))
            tmp_value.append(value[i][j])
            tmp_instance.append(instance[i][j])
            tmp_job.append(job[i][j])
            tmp_id.append(id_[i][j])
    preprocess_data = list(zip(tmp_name,tmp_timestamp,tmp_value,tmp_instance,tmp_job,tmp_id))
    df = pd.DataFrame(preprocess_data, columns=['name','timestamp','value','instance','job','id'])
    return df


# In[47]:


import gc
gc.collect()


# In[48]:


df = create_df_from_dict(name, timestamp, value, instance, job, id_)
# df2 = create_df_from_dict(name2, timestamp2, value2, instance2, job, hostname2, application2, id_2, area2, pool2)


# In[49]:


df


# In[50]:


df.to_csv('prometheus_standardized_data.csv',sep=',',index = 0)
#df2.to_csv('prometheus_standardized_data.csv',sep=',',index = 0)


# In[51]:


read_df = pd.read_csv('prometheus_standardized_data.csv', low_memory=False)
read_df

