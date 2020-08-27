
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pydot as pyd
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization,     multiply, concatenate, Flatten, Activation, dot
from keras.layers import Bidirectional
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

from keras.utils.vis_utils import plot_model, model_to_dot
keras.utils.vis_utils.pydot = pyd

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
scaler = MinMaxScaler()


# In[2]:


# read csv data
origin_df = pd.read_csv('all_metric_data.csv')
origin_pred_df = pd.read_csv('all_metric_data(predict).csv')

reboot_df = pd.read_csv('reboot_time.csv')
predict_reboot_df = pd.read_csv('reboot_time(predict).csv')

# setting sliding window size
minutes = 5
window_parameter = 60*minutes
window_size = int(window_parameter/30)

# columns name in df & predict_df
columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset']

# parameter setting
n_epochs = 3
model_type = 3
filter_out_breakpoint = 'yes'


# In[3]:


origin_pred_df


# In[4]:


# extract all reboot index
reboot = reboot_df['reboot'].values
predict_reboot = predict_reboot_df['reboot'].values

def extract_reboot_time(reboot, window_size):
    reboot_index = []
    for i,v in enumerate(reboot):
        if v == 1:
            reboot_index.append(i-window_size)
    return reboot_index


t1 = extract_reboot_time(reboot, window_size)
t2 = extract_reboot_time(predict_reboot, window_size)    


# In[5]:


# create model input/ouput data
def new_reshape(data):
    tmp = []
    for i in data:
        tmp.append(i[0])
    return tmp

def extract_each_column_v2(column):
    data = origin_df[column].values
    data2 = origin_pred_df[column].values
    scaler.fit(data.reshape(-1,1))
    scaled_data = new_reshape(scaler.transform(data.reshape(-1,1)))
    scaled_data2 = new_reshape(scaler.transform(data2.reshape(-1,1)))
    return data2, scaled_data, scaled_data2 

def data_split(data, window_size):
    history = []
    predict = []
    for i in range(len(data)):
        end_ix = i + window_size
        if end_ix > len(data)-1:
            break
        # i to end_ix as input, end_ix as target output
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        history.append(seq_x)
        predict.append(seq_y)
    return np.array(history), np.array(predict)


# In[6]:


# np.shape example
a = np.zeros([12,1])
print(a)


# In[7]:


# filter out reboot point
def filterout_breakpoint(t,x,y):
    x2,y2 = [],[]
    for i in range(len(x)):
        if i in t:
            continue
        else:
            x2.append(x[i])
            y2.append(y[i])    
    return np.array(x2), np.array(y2)

def plot_compare(time, y1, y2, title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(time, y1, label = 'pred', color='tab:green')
    plt.plot(time, y2, label = 'truth', color='tab:orange')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.show()

def plot_df(time, y, color, re_indx, title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(time, y, color=color)
    plt.plot(re_indx, y[re_indx],'ro')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()   
    
    
def LSTM_model(x_train,y_train,x_test,y_test,origin_x_test,origin_y_test,model_type):
    # lSTM model types:
    if model_type == 1:
        # Single cell LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu',input_shape=(x_train.shape[1],1)))
        model.add(Dense(1))
    if model_type == 2:
        # Stacked LSTM (3 cells)
        model = Sequential()
        model.add(LSTM(128, activation='relu',input_shape=(x_train.shape[1],1), return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(1))
    if model_type == 3:
        # Bidirectional LSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(128, activation='relu'), input_shape=(x_train.shape[1],1)))
        model.add(Dense(1))

    model.compile(optimizer = 'adam',loss = 'mean_absolute_error')
    print(model.summary())

    # model training 
    model.fit(x_train, y_train, validation_split = 0.2, epochs = n_epochs, batch_size = 32 ,verbose = 1)
    print('\n')
    
    # model Evaluation
    y_pred = model.predict(x_test)
    score = mean_absolute_error(y_test,y_pred)
    
    o_y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
    
    time = np.arange(len(y_test))
    plot_df(time, y_test, 'tab:blue', t2, title= 'metric: '+i+' 0727-0729 scaled ground_truth')
    plot_df(time, y_pred[:,0], 'tab:green', t2, title= 'metric: '+i+' 0727-0729 model prediction')
    print('metric: '+i+', score(mean_absoluted_error):',score)

    plot_df(time, origin_y_test,'tab:blue', t2, title= 'metric: '+i+' 0727-0729 model original ground truth ')
    plot_df(time, o_y_pred[:,0],'tab:green', t2, title= 'metric: '+i+' 0727-0729 model inverse prediction')
    plt.show()


# In[8]:


for i in columns:
    origin_test, train, test  = extract_each_column_v2(i)

    # split into moving window
    x_train, y_train = data_split(train, window_size)
    x_test, y_test = data_split(test, window_size)
    
    # filter out reboot point in training
    if filter_out_breakpoint == 'yes':
        x_train, y_train = filterout_breakpoint(t1,x_train,y_train)
        
    # origin
    origin_x_test, origin_y_test = data_split(origin_test, window_size)
    
    # reshape to fit into model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test= np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    origin_x_test = np.reshape(origin_x_test, (origin_x_test.shape[0],origin_x_test.shape[1],1))
    LSTM_model(x_train,y_train,x_test,y_test,origin_x_test,origin_y_test,model_type)

