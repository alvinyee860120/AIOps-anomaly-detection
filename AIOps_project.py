
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


df = pd.read_csv('train_test.csv')
predict_df = pd.read_csv('predict.csv')


reboot_df = pd.read_csv('reboot_time.csv')
predict_reboot_df = pd.read_csv('reboot_time(predict).csv')

# setting sliding window size
minutes = 60
window_parameter = 60*minutes
window_size = int(window_parameter/30)

train_size = int(0.7*df.shape[0])  # number of timestamp to train from
test_size = int(0.3*df.shape[0]) # number of timestamp to be test from

# parameter setting
n_epochs = 5
oversampling_type = 2
label_type = 2
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


count_class_0, count_class_1 = df['label'].value_counts()
print(count_class_0, count_class_1)


# In[8]:


# create model input/ouput data
y = df['label'].values
x_df = df.drop('label',axis=1)
x = x_df[::].values

truth = predict_df['label'].values
predict = predict_df.drop('label',axis = 1)
predict_data = predict[::].values

print(x.shape,type(x),len(x))
print(y.shape,type(y),len(y))
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
            

# create train / test
def create_train_test(df):
    train_set = df[0:train_size]
    test_set = df[train_size: train_size+test_size]
    return train_set, test_set

x_train, x_test = create_train_test(x)
y_train, y_test = create_train_test(y)

if filter_out_breakpoint = 'yes':
    x_train, y_train = filterout_breakpoint(t1,x_train,y_train)

# handle imbalanced dataset by oversampling minority data
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)

if oversampling_type == 1:
    sm = RandomOverSampler(random_state=42)
elif oversampling_type == 2:
    sm = SMOTE(random_state=42)
    
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
cnt1,cnt2,cnt3,cnt4 = 0, 0, 0, 0
for i in y_train_res:
    if i == 0:
        cnt1 += 1
    elif i == 1:
        cnt2 += 1
    elif i == 2:
        cnt3 += 1
    else:
        cnt4 += 1
print(cnt1,cnt2,cnt3,cnt4)


# In[9]:


predict_data


# In[10]:


print(len(x_train_res)) #number of sliding window in train
print(len(x_train_res[0])) # size of each sliding window
print(x_train_res)
print(y_train_res)
print('------------------------------------------------')
print(len(x_test)) #number of sliding window in train
print(len(x_test[0])) # size of each sliding window
print(x_test)
print(y_test)


# In[11]:


# stroe temp value in comparison with y_pred
tmp_y_test = y_test


# In[12]:


x_train_res = np.reshape(x_train_res, (x_train_res.shape[0],1,x_train_res.shape[1]))
x_test= np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))

predict_data =  np.reshape(predict_data, (predict_data.shape[0],1,predict_data.shape[1]))


# In[13]:


predict_data.shape


# In[14]:


# np.shape example
a = np.zeros([2,12])
print(a)


# In[15]:


# lSTM model types:
# each model performs at least 94% in f1-score:
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


# In[16]:


if label_type > 2:
    model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
else:
    model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

y_train_res = np_utils.to_categorical(y_train_res)
y_test = np_utils.to_categorical(y_test)
print(model.summary())


# In[17]:


model.fit(x_train_res, y_train_res, validation_split = 0.2, epochs = n_epochs, batch_size = 32 ,verbose = 1)
score = model.evaluate(x_test,y_test)
print('test loss:', score[0])
print('test accuracy:',score[1])


# In[18]:


# model Evaluation
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score, classification_report
y_pred = model.predict_classes(x_test)
print('check shape is equal or not:')
print(y_pred.shape)
print(tmp_y_test.shape,'\n')
print('confusion matrix:')
print(confusion_matrix(tmp_y_test,y_pred),'\n')
print('performance:')
print(classification_report(tmp_y_test,y_pred))


# In[19]:


# predict 07/27-0729 
pred = model.predict_classes(predict_data)
print(pred.shape)
print(truth.shape,'\n')
print(np.unique(pred))
print(np.unique(truth))



print('confusion matrix:')
print(confusion_matrix(truth,pred),'\n')
print('performance:')
print(classification_report(truth,pred))

# cause overfiiting problem --> 各個時段的離群值有所不同


# In[20]:


time = np.arange(len(truth))


# In[21]:


# Visualization
import matplotlib.pyplot as plt
def masked_plot(time,value,title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,6), dpi=dpi)
    masked = np.ma.masked_equal(value, 1)
    masked2 = np.ma.masked_equal(value, 0)
    plt.plot(masked2, color='tab:green', linewidth=2)
    plt.plot(masked, color= 'red', linewidth=2)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
def plotting(time,value,code,title="", xlabel='Time', ylabel='Value', dpi=200):
    if code == 'truth':
        plt.figure(figsize=(16,8), dpi=dpi)
        plt.scatter(time,value, color='tab:green', linewidth=0.5)
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    elif code == 'pred':
        plt.figure(figsize=(16,8), dpi=dpi)
        plt.scatter(time,value, color='tab:green', linewidth=0.5)
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
    
    
def marker_plot(time,value,title="", xlabel='Time', ylabel='Value', dpi=200):
    
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(time, value, color='tab:orange')
    # Add below threshold markers
    below_threshold = y < 1
    plt.scatter(time[below_threshold], value[below_threshold], color='tab:green') 
    
    # Add above threshold markers
    above_threshold = np.logical_not(below_threshold)
    plt.scatter(time[above_threshold], value[above_threshold], color='r')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[22]:


# masked_plot(time,truth, title= 'mip-svc-push-notify-counter(0727-0729)')
# masked_plot(time,pred, title= 'mip-svc-push-notify-counter(0727-0729)')


# In[23]:


plotting(time,truth, code = 'truth', title= 'mip-svc-push-notify-counter(truth)')
plotting(time,pred, code = 'pred', title= 'mip-svc-push-notify-counter(pred)')

