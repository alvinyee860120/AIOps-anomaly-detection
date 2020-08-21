
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[11]:


def extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11):
    v1, v2 = replace_null(list(df1['value'].values)), replace_null(list(df2['value'].values))
    v3, v4 = replace_null(list(df3['value'].values)), replace_null(list(df4['value'].values))
    v5, v6 = replace_null(list(df5['value'].values)), replace_null(list(df6['value'].values))
    v7, v8 = replace_null(list(df7['value'].values)), replace_null(list(df8['value'].values))
    v9, v10 = replace_null(list(df9['value'].values)), replace_null(list(df10['value'].values))
    v11 = replace_null(list(df11['value'].values))
    return v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11

def replace_null(df):
    new= []
    for i,v in enumerate(df):
        if np.nan_to_num(float(v)) == 0.0:
            new.append(0.0)
        else:
            new.append(v)
    return new
            
def handle_kafka_data(df):
    kafka_v = []
    v = df['value'].values
    for i,val in enumerate(v):
        if i % 2 == 0:
            kafka_v.append(val)
    return kafka_v

def delta_kafka(list1,final):
    new_list = []
    for i in range(len(list1)-1):
        new_list.append(list1[i+1]-list1[i])
    new_list.append(final-list1[-1])
    return new_list


def create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11):
    data = list(zip(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11))
    df = pd.DataFrame(data,columns = ['timestamp'
                                      ,'collections(end of minor GC (Allocatin Failure))'
                                      ,'process_cpu_usage'
                                      ,'memory_usaged PS_Eden_Space'
                                      ,'memory_usaged PS_Old_Gen'
                                      ,'pause durations(avg end of minor GC (Allocation Failure))'
                                      ,'pause durations(max end of minor GC (Allocation Failure))'
                                      ,'thread(live)'
                                      ,'thread(daemon)'
                                      ,'thread(peak)'
                                      ,'tomcat_threads(busy)'
                                      ,'tomcat_threads(current)'])
    return df

def concat_timeseries_data(df1,df2,df3):
    new_df = pd.concat([df1,df2,df3]).reset_index(drop=True)
    return new_df


# In[3]:


df1 = replace_null(handle_kafka_data(pd.read_csv('0727_kafka_df.csv')))
df2 = replace_null(handle_kafka_data(pd.read_csv('0728_kafka_df.csv')))
df3 = replace_null(handle_kafka_data(pd.read_csv('0729_kafka_df.csv')))

kafka = df1+df2+df3

# the last timestamp+1 kafka data
df4 = pd.read_csv('kafka_topic_offset-data-2020-08-17 17_47_35.csv')
val2 = df4['kafka_topic_offset{instance="kafka-monitor-linebc-midlxmsp01.paas.cathaybk.intra.uwccb:80", job="msp-svc-kafka-monitor-LineBC", topic="PUSH-NOTIFY-EVENT"}'].values
print(len(kafka))


# In[14]:


kafka_v = delta_kafka(kafka,val2[0])
kafka_df = pd.DataFrame(kafka_v,columns=['kafka_topic_offset'])


# In[7]:


# all metric in date X
df1 = pd.read_csv('0727_collections_df.csv')
df2 = pd.read_csv('0727_CPU_df.csv')
df3 = pd.read_csv('0727_mem_df.csv')
df4 = pd.read_csv('0727_mem(OldGen)_df.csv')
df5 = pd.read_csv('0727_pause(1)_df.csv')
df6 = pd.read_csv('0727_pause(2)_df.csv')
df7 = pd.read_csv('0727_thread(1ive)_df.csv')
df8 = pd.read_csv('0727_thread(daemon)_df.csv')
df9 = pd.read_csv('0727_thread(peak)_df.csv')
df10 = pd.read_csv('0727_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0727_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values

v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_1 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_1


# In[8]:


# all metric in date x
df1 = pd.read_csv('0728_collections_df.csv')
df2 = pd.read_csv('0728_CPU_df.csv')
df3 = pd.read_csv('0728_mem_df.csv')
df4 = pd.read_csv('0728_mem(OldGen)_df.csv')
df5 = pd.read_csv('0728_pause(1)_df.csv')
df6 = pd.read_csv('0728_pause(2)_df.csv')
df7 = pd.read_csv('0728_thread(1ive)_df.csv')
df8 = pd.read_csv('0728_thread(daemon)_df.csv')
df9 = pd.read_csv('0728_thread(peak)_df.csv')
df10 = pd.read_csv('0728_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0728_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_2 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_2


# In[9]:


# all metric in date x
df1 = pd.read_csv('0729_collections_df.csv')
df2 = pd.read_csv('0729_CPU_df.csv')
df3 = pd.read_csv('0729_mem_df.csv')
df4 = pd.read_csv('0729_mem(OldGen)_df.csv')
df5 = pd.read_csv('0729_pause(1)_df.csv')
df6 = pd.read_csv('0729_pause(2)_df.csv')
df7 = pd.read_csv('0729_thread(1ive)_df.csv')
df8 = pd.read_csv('0729_thread(daemon)_df.csv')
df9 = pd.read_csv('0729_thread(peak)_df.csv')
df10 = pd.read_csv('0729_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0729_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_3 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_3


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0716_collections_df.csv')
df2 = pd.read_csv('0716_CPU_df.csv')
df3 = pd.read_csv('0716_mem_df.csv')
df4 = pd.read_csv('0716_mem(OldGen)_df.csv')
df5 = pd.read_csv('0716_pause(1)_df.csv')
df6 = pd.read_csv('0716_pause(2)_df.csv')
df7 = pd.read_csv('0716_thread(1ive)_df.csv')
df8 = pd.read_csv('0716_thread(daemon)_df.csv')
df9 = pd.read_csv('0716_thread(peak)_df.csv')
df10 = pd.read_csv('0716_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0716_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_4 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_4


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0717_collections_df.csv')
df2 = pd.read_csv('0717_CPU_df.csv')
df3 = pd.read_csv('0717_mem_df.csv')
df4 = pd.read_csv('0717_mem(OldGen)_df.csv')
df5 = pd.read_csv('0717_pause(1)_df.csv')
df6 = pd.read_csv('0717_pause(2)_df.csv')
df7 = pd.read_csv('0717_thread(1ive)_df.csv')
df8 = pd.read_csv('0717_thread(daemon)_df.csv')
df9 = pd.read_csv('0717_thread(peak)_df.csv')
df10 = pd.read_csv('0717_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0717_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_5 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_5


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0720_collections_df.csv')
df2 = pd.read_csv('0720_CPU_df.csv')
df3 = pd.read_csv('0720_mem_df.csv')
df4 = pd.read_csv('0720_mem(OldGen)_df.csv')
df5 = pd.read_csv('0720_pause(1)_df.csv')
df6 = pd.read_csv('0720_pause(2)_df.csv')
df7 = pd.read_csv('0720_thread(1ive)_df.csv')
df8 = pd.read_csv('0720_thread(daemon)_df.csv')
df9 = pd.read_csv('0720_thread(peak)_df.csv')
df10 = pd.read_csv('0720_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0720_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_6 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_6


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0721_collections_df.csv')
df2 = pd.read_csv('0721_CPU_df.csv')
df3 = pd.read_csv('0721_mem_df.csv')
df4 = pd.read_csv('0721_mem(OldGen)_df.csv')
df5 = pd.read_csv('0721_pause(1)_df.csv')
df6 = pd.read_csv('0721_pause(2)_df.csv')
df7 = pd.read_csv('0721_thread(1ive)_df.csv')
df8 = pd.read_csv('0721_thread(daemon)_df.csv')
df9 = pd.read_csv('0721_thread(peak)_df.csv')
df10 = pd.read_csv('0721_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0721_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_7 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_7


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0722_collections_df.csv')
df2 = pd.read_csv('0722_CPU_df.csv')
df3 = pd.read_csv('0722_mem_df.csv')
df4 = pd.read_csv('0722_mem(OldGen)_df.csv')
df5 = pd.read_csv('0722_pause(1)_df.csv')
df6 = pd.read_csv('0722_pause(2)_df.csv')
df7 = pd.read_csv('0722_thread(1ive)_df.csv')
df8 = pd.read_csv('0722_thread(daemon)_df.csv')
df9 = pd.read_csv('0722_thread(peak)_df.csv')
df10 = pd.read_csv('0722_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0722_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_8 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_8


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0723_collections_df.csv')
df2 = pd.read_csv('0723_CPU_df.csv')
df3 = pd.read_csv('0723_mem_df.csv')
df4 = pd.read_csv('0723_mem(OldGen)_df.csv')
df5 = pd.read_csv('0723_pause(1)_df.csv')
df6 = pd.read_csv('0723_pause(2)_df.csv')
df7 = pd.read_csv('0723_thread(1ive)_df.csv')
df8 = pd.read_csv('0723_thread(daemon)_df.csv')
df9 = pd.read_csv('0723_thread(peak)_df.csv')
df10 = pd.read_csv('0723_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0723_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_9 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_9


# In[ ]:


# all metric in date x
df1 = pd.read_csv('0724_collections_df.csv')
df2 = pd.read_csv('0724_CPU_df.csv')
df3 = pd.read_csv('0724_mem_df.csv')
df4 = pd.read_csv('0724_mem(OldGen)_df.csv')
df5 = pd.read_csv('0724_pause(1)_df.csv')
df6 = pd.read_csv('0724_pause(2)_df.csv')
df7 = pd.read_csv('0724_thread(1ive)_df.csv')
df8 = pd.read_csv('0724_thread(daemon)_df.csv')
df9 = pd.read_csv('0724_thread(peak)_df.csv')
df10 = pd.read_csv('0724_tomcat_threads(busy)_df.csv')
df11 = pd.read_csv('0724_tomcat_threads(cur)_df.csv')
timestamp = df1['time'].values
v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
df_10 = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
df_10


# In[12]:


df = concat_timeseries_data(df_1,df_2,df_3)


# In[15]:


new_df = pd.concat([df,kafka_df],axis=1)
new_df


# In[16]:


df1 = pd.read_csv('0727_reboot_df.csv')['value'].values
df2 = pd.read_csv('0728_reboot_df.csv')['value'].values
df3 = pd.read_csv('0729_reboot_df.csv')['value'].values

reboot = list(df1)+list(df2)+list(df3)
print(len(reboot))
reboot_df = pd.DataFrame(reboot,columns = ['reboot_count'])


# In[17]:


new_df = pd.concat([new_df,reboot_df],axis=1)
new_df


# In[18]:


new_df.to_csv('all_metric_data(predict).csv',sep=',',index = 0)

