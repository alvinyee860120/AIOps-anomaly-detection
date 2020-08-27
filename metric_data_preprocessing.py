
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# setting label type
label_type = 3

# # setting offset window size
# offset_time_limit = 1*60
# offset_time_limit_size = int(offset_time_limit/30)


# In[3]:


df = pd.read_csv('all_metric_data.csv')
df2 = pd.read_csv('all_metric_data(predict).csv')


# In[4]:


time = df['timestamp'].values
collection = df['collections(end of minor GC (Allocatin Failure))'].values
cpu = df['process_cpu_usage'].values
mem_eden = df['memory_usaged PS_Eden_Space'].values
mem_old = df['memory_usaged PS_Old_Gen'].values
pause_avg = df['pause durations(avg end of minor GC (Allocation Failure))'].values
pause_max = df['pause durations(max end of minor GC (Allocation Failure))'].values
thread_live = df['thread(live)'].values
thread_daemon = df['thread(daemon)'].values
thread_peak =df['thread(peak)'].values
tomcat_busy = df['tomcat_threads(busy)'].values
tomcat_cur = df['tomcat_threads(current)'].values
offset = df['kafka_topic_offset'].values
reboot = df['reboot_count'].values


# In[5]:


def compute_normal_and_outliers(array):
    temp = []
    for i in array:
        if i != 0 and i > 0:
            temp.append(i)
    mean = np.mean(temp)
    std = np.std(temp)
    final = []
    print(mean,std)
    for i in array:
        if i == 0 or i < 0:
            final.append(1)
        elif i > mean+2*std or i < mean-2*std :
            final.append(1)
        else:
            final.append(0)
    upper_bound, lower_bound = mean+2*std, mean-2*std
    print(upper_bound,lower_bound)
    count = 0
    for i in final:
        if i == 1:
            count+=1
    print(count)
    print('------------')
    return final, upper_bound, lower_bound

def compute_normal_and_outliers2predict(array,upper,lower):
    temp = []
    for i in array:
        if i != 0 and i > 0:
            temp.append(i)
    final = []
    for i in array:
        if i == 0 or i < 0:
            final.append(1)
        elif i > upper or i < lower:
            final.append(1)
        else:
            final.append(0)
    count = 0
    for i in final:
        if i == 1:
            count+=1
    print(count)
    
    return final

# def check_offest_interval_anomaly(array,offset_time_limit_size):
#     threshold = 100
#     final = []
#     count = 0
#     for i in range(len(array),offset_time_limit_size):
#         if sum(array[i:i+offset_time_limit_size]) < offset_time_limit_size*threshold:
#             final[i:i+offset_time_limit_size] = 1
#             count += offset_time_limit_size
#         else:
#             final[i:i+offset_time_limit_size] = 0
#     print(count)
#     return final


# In[6]:


# compute outliers & normal of each metric(excluding null point)
collection_,upper1,lower1 = compute_normal_and_outliers(collection)
cpu_,upper2,lower2 = compute_normal_and_outliers(cpu)
mem_eden_,upper3,lower3 = compute_normal_and_outliers(mem_eden)
mem_old_,upper4,lower4 = compute_normal_and_outliers(mem_old)
pause_avg_,upper5,lower5 = compute_normal_and_outliers(pause_avg)
pause_max_,upper6,lower6 = compute_normal_and_outliers(pause_max)
thread_live_,upper7,lower7 = compute_normal_and_outliers(thread_live)
thread_daemon_,upper8,lower8 = compute_normal_and_outliers(thread_daemon)
thread_peak_,upper9,lower9 = compute_normal_and_outliers(thread_peak)
tomcat_busy_,upper10,lower10 = compute_normal_and_outliers(tomcat_busy)
tomcat_cur_,upper11,lower11 = compute_normal_and_outliers(tomcat_cur)
offset_,upper12,lower12 = compute_normal_and_outliers(offset)

# setting value of delta offset below threshold as anomaly, otherwise normal 
# offset2 = check_offest_interval_anomaly(offset,offset_time_limit_size)


# In[7]:


# create one hot table 
columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset']
df_ = list(zip(collection_,cpu_,mem_eden_,mem_old_,pause_avg_,pause_max_,thread_live_
         ,thread_daemon_,thread_peak_,tomcat_busy_,tomcat_cur_,offset_))
df_ = pd.DataFrame(df_,columns = columns) 
df_


# In[8]:


#binary labeling:
if label_type == 2:
    label = []
    alldf = df_[::].values
    print(alldf.shape)
    print(type(alldf))
    print(alldf)

    # 0 for normal, 1 for abnormal
    for i in alldf:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        if cnt > 0:
            label.append(1)   # anomaly
        else:
            label.append(0)   # normal

    print(len(label))
    print(label.count(0))
    print(label.count(1))
    print('normal percentage: ',label.count(0)/len(label)*100,'%')



#trinary labeling:
if label_type == 3:
    label = []
    alldf = df_[::].values
    print(alldf.shape)
    print(type(alldf))
    print(alldf)

    # 0 for normal, 1 for abnormal
    for i in alldf:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        if cnt >= 8:
            label.append(2)   # high risk
        elif cnt >= 4 and cnt < 8:
            label.append(1)   # middle risk
        elif cnt < 4:
            label.append(0)   # low risk

    print(len(label))
    print(label.count(0))
    print(label.count(1))
    print(label.count(2))
    print('normal percentage: ',label.count(0)/len(label)*100,'%')


# In[9]:


def new_reshape(data):
    tmp = []
    for i in data:
        tmp.append(i[0])
    return tmp


# In[10]:


# reshape example
a = collection.reshape(-1,1)
print(collection[0])
print(a[0])
print(collection.shape)
print(a.shape)


# In[11]:


def scaler_trans(train,test):
    # scaler
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_pred = scaler.transform(test)
    return scaled_train, scaled_pred

# 0727 -0729
time2 = df2['timestamp'].values
collection2 = df2['collections(end of minor GC (Allocatin Failure))'].values
cpu2 = df2['process_cpu_usage'].values
mem_eden2 = df2['memory_usaged PS_Eden_Space'].values
mem_old2 = df2['memory_usaged PS_Old_Gen'].values
pause_avg2 = df2['pause durations(avg end of minor GC (Allocation Failure))'].values
pause_max2 = df2['pause durations(max end of minor GC (Allocation Failure))'].values
thread_live2 = df2['thread(live)'].values
thread_daemon2 = df2['thread(daemon)'].values
thread_peak2 =df2['thread(peak)'].values
tomcat_busy2 = df2['tomcat_threads(busy)'].values
tomcat_cur2 = df2['tomcat_threads(current)'].values
offset2 = df2['kafka_topic_offset'].values
reboot2 = df2['reboot_count'].values

# construct predict label
collection_lbl = compute_normal_and_outliers2predict(collection2,upper1,lower1)
cpu_lbl = compute_normal_and_outliers2predict(cpu2,upper2,lower2)
mem_eden_lbl = compute_normal_and_outliers2predict(mem_eden2,upper3,lower3)
mem_old_lbl = compute_normal_and_outliers2predict(mem_old2,upper4,lower4)
pause_avg_lbl = compute_normal_and_outliers2predict(pause_avg2,upper5,lower5)
pause_max_lbl = compute_normal_and_outliers2predict(pause_max2,upper6,lower6)
thread_live_lbl = compute_normal_and_outliers2predict(thread_live2,upper7,lower7)
thread_daemon_lbl = compute_normal_and_outliers2predict(thread_daemon2,upper8,lower8)
thread_peak_lbl = compute_normal_and_outliers2predict(thread_peak2,upper9,lower9)
tomcat_busy_lbl = compute_normal_and_outliers2predict(tomcat_busy2,upper10,lower10)
tomcat_cur_lbl = compute_normal_and_outliers2predict(tomcat_cur2,upper11,lower11)
offset_lbl = compute_normal_and_outliers2predict(offset2,upper12,lower12)


# normalization
collection_train, collection_pred = scaler_trans(collection.reshape(-1,1),collection2.reshape(-1,1))
cpu_train, cpu_pred = scaler_trans(cpu.reshape(-1,1),cpu2.reshape(-1,1))
mem_eden_train, mem_eden_pred = scaler_trans(mem_eden.reshape(-1,1),mem_eden2.reshape(-1,1))
mem_old_train, mem_old_pred = scaler_trans(mem_old.reshape(-1,1),mem_old2.reshape(-1,1))
pause_avg_train, pause_avg_pred = scaler_trans(pause_avg.reshape(-1,1),pause_avg2.reshape(-1,1))
pause_max_train, pause_max_pred = scaler_trans(pause_max.reshape(-1,1),pause_max2.reshape(-1,1))
thread_live_train, thread_live_pred = scaler_trans(thread_live.reshape(-1,1),thread_live2.reshape(-1,1))
thread_daemon_train, thread_daemon_pred = scaler_trans(thread_daemon.reshape(-1,1),thread_daemon2.reshape(-1,1))
thread_peak_train, thread_peak_pred = scaler_trans(thread_peak.reshape(-1,1),thread_peak2.reshape(-1,1))                                   
tomcat_busy_train, tomcat_busy_pred = scaler_trans(tomcat_busy.reshape(-1,1),tomcat_busy2.reshape(-1,1))                                   
tomcat_cur_train, tomcat_cur_pred = scaler_trans(tomcat_cur.reshape(-1,1),tomcat_cur2.reshape(-1,1))                                  
offset_train, offset_pred = scaler_trans(offset.reshape(-1,1),offset2.reshape(-1,1))  
                                                 
                                                 
                                                 
# reshape
# train
collection = new_reshape(collection_train)                                                 
cpu = new_reshape(cpu_train)
mem_eden = new_reshape(mem_eden_train)
mem_old = new_reshape(mem_old_train)
pause_avg = new_reshape(pause_avg_train)
pause_max = new_reshape(pause_max_train)
thread_live = new_reshape(thread_live_train)
thread_daemon = new_reshape(thread_daemon_train)
thread_peak = new_reshape(thread_peak_train)
tomcat_busy = new_reshape(tomcat_busy_train)
tomcat_cur = new_reshape(tomcat_cur_train)
offset = new_reshape(offset_train)

# predict
collection3 = new_reshape(collection_pred)                                                 
cpu3 = new_reshape(cpu_pred)
mem_eden3 = new_reshape(mem_eden_pred)
mem_old3 = new_reshape(mem_old_pred)
pause_avg3 = new_reshape(pause_avg_pred)
pause_max3 = new_reshape(pause_max_pred)
thread_live3 = new_reshape(thread_live_pred)
thread_daemon3 = new_reshape(thread_daemon_pred)
thread_peak3 = new_reshape(thread_peak_pred)
tomcat_busy3 = new_reshape(tomcat_busy_pred)
tomcat_cur3 = new_reshape(tomcat_cur_pred)
offset3 = new_reshape(offset_pred)                          


# In[12]:


# scaled training data
columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset','label']
df_train= list(zip(collection,cpu,mem_eden,mem_old,pause_avg,pause_max,thread_live
         ,thread_daemon,thread_peak,tomcat_busy,tomcat_cur,offset,label))
df_train = pd.DataFrame(df_train,columns = columns)
df_train.to_csv('train_test.csv',sep=',',index = 0)
df_train


# In[13]:


# scaled testing data
columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset']
df_pred = list(zip(collection3,cpu3,mem_eden3,mem_old3,pause_avg3,pause_max3,thread_live3
         ,thread_daemon3,thread_peak3,tomcat_busy3,tomcat_cur3,offset3))
df_pred = pd.DataFrame(df_pred,columns = columns)
df_pred


# In[14]:


# one-hot label for predict data
columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset']
df_predlbl = list(zip(collection_lbl,cpu_lbl,mem_eden_lbl,mem_old_lbl,pause_avg_lbl,pause_max_lbl,thread_live_lbl
         ,thread_daemon_lbl,thread_peak_lbl,tomcat_busy_lbl,tomcat_cur_lbl,offset_lbl))
df_predlbl = pd.DataFrame(df_predlbl,columns = columns)
df_predlbl


# In[15]:


#binary labeling:
if label_type == 2:
    label = []
    alldf = df_predlbl[::].values
    print(alldf.shape)
    print(type(alldf))
    print(alldf)

    # 0 for normal, 1 for abnormal
    for i in alldf:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        if cnt > 0:
            label.append(1)   # anomaly
        else:
            label.append(0)   # normal

    print(len(label))
    print(label.count(0))
    print(label.count(1))
    print('normal percentage: ',label.count(0)/len(label)*100,'%')



#trinary labeling:
if label_type == 3:
    label = []
    alldf = df_predlbl[::].values
    print(alldf.shape)
    print(type(alldf))
    print(alldf)

    # 0 for normal, 1 for abnormal
    for i in alldf:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        if cnt >= 8:
            label.append(2)   # high risk
        elif cnt >= 4 and cnt < 8:
            label.append(1)   # middle risk
        elif cnt < 4:
            label.append(0)   # low risk

    print(len(label))
    print(label.count(0))
    print(label.count(1))
    print(label.count(2))
    print('normal percentage: ',label.count(0)/len(label)*100,'%')

# four label labeling:
if label_type == 4:
    label = []
    alldf = df_predlbl[::].values
    print(alldf.shape)
    print(type(alldf))
    print(alldf)

    # 0 for normal, 1 for abnormal
    for i in alldf:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        if cnt >= 8:
            label.append(3)   # high risk
        elif cnt >= 4 and cnt < 8:
            label.append(2)   # middle risk
        elif cnt < 4 and cnt > 0:
            label.append(1)   # low risk
        else:
            label.append(0)   # no risk

    print(len(label))
    print(label.count(0))
    print(label.count(1))
    print(label.count(2))
    print(label.count(3))
    print('normal percentage: ',label.count(0)/len(label)*100,'%')


label_df = pd.DataFrame(label,columns = ['label'])
new_df = pd.concat([df_pred,label_df],axis=1)
new_df.to_csv('predict.csv',sep=',',index = 0)


# In[16]:


# create reboot csv file
count = 0
for i in reboot:
    if i == 1:
        count += 1
print(count)
        
count2 = 0
for i in reboot2:
    if i == 1:
        count2 +=1
print(count2)


reboot_df = pd.DataFrame(reboot,columns = ['reboot'])
reboot_df.to_csv('reboot_time.csv',sep=',',index = 0)

reboot_df2 = pd.DataFrame(reboot2,columns = ['reboot'])
reboot_df2.to_csv('reboot_time(predict).csv',sep=',',index = 0)

