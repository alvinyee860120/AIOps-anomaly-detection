
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


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
        if i % 2 == 0 or i == 0:
            kafka_v.append(val)
    return kafka_v

def delta_kafka(list1,final):
    new_list = []
    for i in range(len(list1)-1):
        print(i)
        new_list.append(list1[i+1]-list1[i])
    print(final-list1[-1])
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


def create_daily_all_metric_data(date):
    df1 = pd.read_csv(date+'_collections_df.csv')
    df2 = pd.read_csv(date+'_CPU_df.csv')
    df3 = pd.read_csv(date+'_mem_df.csv')
    df4 = pd.read_csv(date+'_mem(OldGen)_df.csv')
    df5 = pd.read_csv(date+'_pause(1)_df.csv')
    df6 = pd.read_csv(date+'_pause(2)_df.csv')
    df7 = pd.read_csv(date+'_thread(1ive)_df.csv')
    df8 = pd.read_csv(date+'_thread(daemon)_df.csv')
    df9 = pd.read_csv(date+'_thread(peak)_df.csv')
    df10 = pd.read_csv(date+'_tomcat_threads(busy)_df.csv')
    df11 = pd.read_csv(date+'_tomcat_threads(cur)_df.csv')
    timestamp = df1['time'].values

    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11 = extract_values(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11)
    df = create_csv(timestamp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11)
    return df


def concat_timeseries_data(df1,df2,df3):
    new_df = pd.concat([df1,df2,df3]).reset_index(drop=True)
    return new_df


# In[ ]:


df_1 = create_daily_all_metric_data('0713')
df_2 = create_daily_all_metric_data('0714')
df_3 = create_daily_all_metric_data('0715')
df_4 = create_daily_all_metric_data('0716')
df_5 = create_daily_all_metric_data('0717')
df_6 = create_daily_all_metric_data('0720')
df_7 = create_daily_all_metric_data('0721')
df_8 = create_daily_all_metric_data('0722')
df_9 = create_daily_all_metric_data('0723')
df_10 = create_daily_all_metric_data('0724')
df_11 = create_daily_all_metric_data('0727')
df_12 = create_daily_all_metric_data('0728')
df_13 = create_daily_all_metric_data('0729')


# In[ ]:


# concat training kafka data
df1 = replace_null(handle_kafka_data(pd.read_csv('0713_kafka_df.csv')))
df2 = replace_null(pd.read_csv('0714_kafka_df.csv'))
df3 = replace_null(pd.read_csv('0715_kafka_df.csv'))
df4 = replace_null(pd.read_csv('0716_kafka_df.csv'))
df5 = replace_null(pd.read_csv('0717_kafka_df.csv'))
df6 = replace_null(pd.read_csv('0720_kafka_df.csv'))
df7 = replace_null(pd.read_csv('0721_kafka_df.csv'))
df8 = replace_null(pd.read_csv('0722_kafka_df.csv'))
df9 = replace_null(pd.read_csv('0723_kafka_df.csv'))
df10 = replace_null(pd.read_csv('0724_kafka_df.csv'))

kafka1 = df1+df2+df3+df4+df5
kafka2 = df6+df7+df8+df9+df10
print(len(kafka1))
print(len(kafka2))

# the last timestamp+1 kafka data
df_k1 = pd.read_csv('kafka_topic_offset-data-07180000000.csv')
df_k2 = pd.read_csv('kafka_topic_offset-data-07250000000.csv')
val1 = df_k1['kafka_topic_offset{instance="kafka-monitor-linebc-midlxmsp01.paas.cathaybk.intra.uwccb:80", job="msp-svc-kafka-monitor-LineBC", topic="PUSH-NOTIFY-EVENT"}'].values
val2 = df_k2['kafka_topic_offset{instance="kafka-monitor-linebc-midlxmsp01.paas.cathaybk.intra.uwccb:80", job="msp-svc-kafka-monitor-LineBC", topic="PUSH-NOTIFY-EVENT"}'].values

kafka_v1 = delta_kafka(kafka,val1[0])
kafka_v2 = delta_kafka(kafka,val2[0])

kafka_v = kafka_v1+kafka_v2
kafka_df = pd.DataFrame(kafka_v,columns=['kafka_topic_offset'])


# In[ ]:


# concat testing kafka data
df1 = replace_null(handle_kafka_data(pd.read_csv('0727_kafka_df.csv')))
df2 = replace_null(handle_kafka_data(pd.read_csv('0728_kafka_df.csv')))
df3 = replace_null(handle_kafka_data(pd.read_csv('0729_kafka_df.csv')))

kafka = df1+df2+df3
print(len(kafka))

# the last timestamp+1 kafka data
df_k3 = pd.read_csv('kafka_topic_offset-data-2020-08-17 17_47_35.csv')
val3 = df_k3['kafka_topic_offset{instance="kafka-monitor-linebc-midlxmsp01.paas.cathaybk.intra.uwccb:80", job="msp-svc-kafka-monitor-LineBC", topic="PUSH-NOTIFY-EVENT"}'].values
kafka_v = delta_kafka(kafka,val3[0])
kafka_df2 = pd.DataFrame(kafka_v,columns=['kafka_topic_offset'])


# In[ ]:


train_df = concat_timeseries_data(df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10)
test_df = concat_timeseries_data(df_11,df_12,df_13)


# In[ ]:


df1 = pd.read_csv('0713_reboot_df.csv')['value'].values
df2 = pd.read_csv('0714_reboot_df.csv')['value'].values
df3 = pd.read_csv('0715_reboot_df.csv')['value'].values
df4 = pd.read_csv('0716_reboot_df.csv')['value'].values
df5 = pd.read_csv('0717_reboot_df.csv')['value'].values
df6 = pd.read_csv('0720_reboot_df.csv')['value'].values
df7 = pd.read_csv('0721_reboot_df.csv')['value'].values
df8 = pd.read_csv('0722_reboot_df.csv')['value'].values
df9 = pd.read_csv('0723_reboot_df.csv')['value'].values
df10 = pd.read_csv('0724_reboot_df.csv')['value'].values
reboot = list(df1)+list(df2)+list(df3)+list(df4)+list(df5)+list(df6)+list(df7)+list(df8)+list(df9)+list(df10)
print(len(reboot))
reboot_df = pd.DataFrame(reboot,columns = ['reboot_count'])


# In[ ]:


newtrain_df = pd.concat([train_df,kafka_df],axis=1)
newtrain_df = pd.concat([newtrain_df,reboot_df],axis=1)
newtrain_df


# In[ ]:


newtest_df.to_csv('all_metric_data.csv',sep=',',index = 0)


# In[ ]:


df1 = pd.read_csv('0727_reboot_df.csv')['value'].values
df2 = pd.read_csv('0728_reboot_df.csv')['value'].values
df3 = pd.read_csv('0729_reboot_df.csv')['value'].values

reboot = list(df1)+list(df2)+list(df3)
print(len(reboot))
reboot_df2 = pd.DataFrame(reboot,columns = ['reboot_count'])


# In[ ]:


newtest_df = pd.concat([test_df,kafka_df2],axis=1)
newtest_df = pd.concat([newtest_df,reboot_df2],axis=1)
newtest_df


# In[ ]:


newtest_df.to_csv('all_metric_data(predict).csv',sep=',',index = 0)

