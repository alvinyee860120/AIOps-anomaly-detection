
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.utils import np_utils
import keras


# In[2]:


# read training data
df = pd.read_csv('train_test.csv')
predict_df = pd.read_csv('predict.csv')

# read testing data
reboot_df = pd.read_csv('reboot_time.csv')
predict_reboot_df = pd.read_csv('reboot_time(predict).csv')

# parameter setting
n_epochs = 5
oversampling_type = 2
label_type = 3
model_type = 3
filter_out_breakpoint = 'yes'


# In[3]:


predict_df


# In[4]:


# extract all reboot index
reboot = reboot_df['reboot'].values
predict_reboot = predict_reboot_df['reboot'].values

def extract_reboot_time(reboot):
    reboot_index = []
    for i,v in enumerate(reboot):
        if v == 1:
            reboot_index.append(i)
    return reboot_index

t1 = extract_reboot_time(reboot)
t2 = extract_reboot_time(predict_reboot)                


# In[5]:


# 各類別label個數 
count_class_0, count_class_1 = df['label'].value_counts()
print(count_class_0, count_class_1)


# In[6]:


# create model input/ouput data
y = df['label'].values
x_df = df.drop('label',axis=1)
x = x_df[::].values
print(x.shape,type(x),len(x))
print(y.shape,type(y),len(y))

truth = predict_df['label'].values
predict = predict_df.drop('label',axis = 1)
predict_data = predict[::].values
print(predict_data.shape,type(predict_data),len(predict_data))
print(truth.shape,type(truth),len(truth))


# filter out break point
def filterout_breakpoint(t,x,y):
    x2,y2 = [],[]
    for i in range(len(x)):
        if i in t:
            continue
        else:
            x2.append(x[i])
            y2.append(y[i])
    
    return np.array(x2), np.array(y2) 
            

x_train, y_train = x, y
x_test, y_test = predict_data, truth


# filter out breakpoint in training 
if filter_out_breakpoint == 'yes':
    x_train, y_train = filterout_breakpoint(t1,x_train,y_train)
    

    
# handle imbalanced dataset by oversampling minority data
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)

if oversampling_type == 1:
    sm = RandomOverSampler(random_state=42)
elif oversampling_type == 2:
    sm = SMOTE(random_state=42)
    
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
cnt1,cnt2, cnt3 = 0, 0, 0 
for i in y_train_res:
    if i == 0:
        cnt1 += 1
    elif i == 1:
        cnt2 += 1
    else:
        cnt3 += 1
print(cnt1,cnt2,cnt3)


# In[7]:


predict_data


# In[8]:


print(len(x_train_res)) #number of sliding window in train
print(len(x_train_res[0])) # size of each sliding window
print(x_train_res)
print(y_train_res)
print('------------------------------------------------')
print(len(x_test)) #number of sliding window in train
print(len(x_test[0])) # size of each sliding window
print(x_test)
print(y_test)


# In[9]:


# stroe temp value in comparison with y_pred
tmp_y_test = y_test


# In[10]:


# reshape input 
x_train_res = np.reshape(x_train_res, (x_train_res.shape[0],1,x_train_res.shape[1]))
x_test= np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))

predict_data =  np.reshape(predict_data, (predict_data.shape[0],1,predict_data.shape[1]))
predict_data.shape


# In[11]:


# oversampling後的data數
x_train_res.shape


# In[12]:


# np.shape example
a = np.zeros([2,12])
print(a)


# In[13]:


# lSTM model types:
# each model performs at least 90% in f1-score:
if model_type == 1:
    # Single cell LSTM
    model = Sequential()
    model.add(LSTM(128,input_shape=(1, 12)))
    model.add(Dense(label_type,activation = 'sigmoid'))
if model_type == 2:
    # Stacked LSTM (3 cells)
    model = Sequential()
    model.add(LSTM(128,input_shape=(1, 12), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(label_type, activation = 'sigmoid'))
if model_type == 3:
    # Bidirectional LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(1,12)))
    model.add(Dense(label_type,activation = 'sigmoid'))


# In[14]:


if label_type > 2:
    model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
else:
    model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

y_train_res = np_utils.to_categorical(y_train_res)
y_test = np_utils.to_categorical(y_test)
print(model.summary())


# In[15]:


model.fit(x_train_res, y_train_res, validation_split = 0.2, epochs = n_epochs, batch_size = 32 ,verbose = 1)
score = model.evaluate(x_test,y_test)
print('test loss:', score[0])
print('test accuracy:',score[1])


# In[16]:


# model Evaluation
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score, classification_report
y_pred = model.predict_classes(x_test)
print('Check shape is equal or not:')
print(y_pred.shape)
print(tmp_y_test.shape,'\n')
print('Confusion matrix:')
print(confusion_matrix(tmp_y_test,y_pred),'\n')
print('Model performance:')
print(classification_report(tmp_y_test,y_pred))


# In[17]:


# Visualization
import matplotlib.pyplot as plt
def plotting(time,value,code,re_indx,title="", xlabel='Time', ylabel='Value', dpi=200):
    if code == 'truth':
        plt.figure(figsize=(16,8), dpi=dpi)
        plt.plot(time,value, color='tab:blue', linewidth=0.5)
        plt.plot(re_indx,value[re_indx],'ks')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    elif code == 'pred':
        plt.figure(figsize=(16,8), dpi=dpi)
        plt.plot(time,value, color='tab:green', linewidth=0.5)
        plt.plot(re_indx,value[re_indx],'ks')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[18]:


time = np.arange(len(truth))
plotting(time,tmp_y_test, code = 'truth',re_indx=t2,title= 'mip-svc-push-notify-counter(truth)')
plotting(time,y_pred, code = 'pred',re_indx=t2,title= 'mip-svc-push-notify-counter(pred)')


# In[ ]:


# masked_plot(time,truth, title= 'mip-svc-push-notify-counter(0727-0729)')
# masked_plot(time,pred, title= 'mip-svc-push-notify-counter(0727-0729)')

