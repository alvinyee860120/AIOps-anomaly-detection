
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


df = pd.read_csv('train_test.csv')
predict_df = pd.read_csv('predict.csv')

reboot_df = pd.read_csv('reboot_time.csv')
predict_reboot_df = pd.read_csv('reboot_time(predict).csv')

origin_df = pd.read_csv('all_metric_data.csv')
origin_pred_df = pd.read_csv('all_metric_data(predict).csv')


# setting sliding window size
minutes = 5
window_parameter = 60*minutes
window_size = int(window_parameter/30)

columns = ['collections(end of minor GC (Allocatin Failure))','process_cpu_usage'
            ,'memory_usaged PS_Eden_Space','memory_usaged PS_Old_Gen'
           ,'pause durations(avg end of minor GC (Allocation Failure))'
           ,'pause durations(max end of minor GC (Allocation Failure))'
           ,'thread(live)','thread(daemon)','thread(peak)','tomcat_threads(busy)'
            ,'tomcat_threads(current)','kafka_topic_offset']

n_epochs = 3
model_type = 3
version = 2


# In[3]:


df


# In[4]:


# extract reboot index
reboot = reboot_df['reboot'].values
reboot_p = predict_reboot_df['reboot'].values

reboot_index = []
reboot_p_index = []

for i,v in enumerate(reboot):
    if v == 1:
        reboot_index.append(i)

for i,v in enumerate(reboot_p):
    if v == 1:
        reboot_p_index.append(i)


# In[5]:


# count_class_0, count_class_1 = df['label'].value_counts()
# print(count_class_0, count_class_1)


# In[6]:


# create model input/ouput data
# create train / test
def create_train_test(df,train_size,test_size):
    train_set = df[0:train_size]
    test_set = df[train_size: train_size+test_size]
    return train_set, test_set

def new_reshape(data):
    tmp = []
    for i in data:
        tmp.append(i[0])
    return tmp

def extract_each_column(column):
    data = df[column].values
    data2 = predict_df[column].values
    train_size = int(0.7*data.shape[0])  # number of timestamp to train from
    test_size = int(0.3*data.shape[0]) # number of timestamp to test from
    train,test = create_train_test(data,train_size,test_size)
    return train, test, data2

def extract_each_column_v2(column):
    data = origin_df[column].values
    data2 = origin_pred_df[column].values
    scaler.fit(data.reshape(-1,1))
    scaled_data = new_reshape(scaler.transform(data.reshape(-1,1)))
    scaled_data2 = new_reshape(scaler.transform(data2.reshape(-1,1)))
    train_size = int(0.7*data.shape[0])  # number of timestamp to train from
    test_size = int(0.3*data.shape[0]) # number of timestamp to test from
    train,test = create_train_test(scaled_data,train_size,test_size)
    inverse_train, inverse_test = create_train_test(data,train_size,test_size)
    return train, test, inverse_test, data2, scaled_data2

def data_split(data, window_size):
    history = []
    predict = []
    for i in range(len(data)):
        end_ix = i + window_size
        if end_ix > len(data)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        history.append(seq_x)
        predict.append(seq_y)
    return np.array(history), np.array(predict)


# In[10]:


# np.shape example
a = np.zeros([12,1])
print(a)


# In[8]:


def plot_compare(time, y1, y2, title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(time, y1, label = 'pred', color='tab:green')
    plt.plot(time, y2, label = 'truth', color='tab:orange')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.show()

def plot_df(time, y, color, title="", xlabel='Time', ylabel='Value', dpi=200):
    plt.figure(figsize=(16,8), dpi=dpi)
    plt.plot(time, y, color=color)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()   
    
    
def LSTM_model(x_train,y_train,x_test,y_test,pred_history,pred_prediction,origin_pred_history,origin_pred_prediction,model_type,version):
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
    if version == 1:
#         y_pred = model.predict(x_test)
#         score = mean_absolute_error(y_test,y_pred)
#         print(score)
#         time = np.arange(len(y_test))
#         plot_compare(time, y_pred[:,0], y_test, title= 'metric: '+i)

        p_pred = model.predict(pred_history)
        score2 = mean_absolute_error(pred_prediction,p_pred)
       
        time2 = np.arange(len(pred_prediction))
        plot_df(time2, pred_prediction,'tab:green', title= 'metric: '+i+' 0727-0729 ground_truth')
        plot_df(time2, p_pred[:,0],'tab:blue', title= 'metric: '+i+' 0727-0729 model prediction')
        print('score(mean_absoluted_error):',score2)
        
#         return p_pred
    
    else:
#         y_pred = model.predict(x_test)
#         score = mean_absolute_error(y_test,y_pred)
#         print(score)
#         time = np.arange(len(y_test))
#         plot_compare(time, y_pred[:,0], y_test, title= 'metric: '+i)

        p_pred = model.predict(pred_history)
        o_p_pred = model.predict(origin_pred_history)
        score2 = mean_absolute_error(pred_prediction,p_pred)
        score3 = mean_absolute_error(origin_pred_prediction,o_p_pred)
        
        
        time2 = np.arange(len(pred_prediction))
        plot_df(time2, pred_prediction, 'tab:blue', title= 'metric: '+i+' 0727-0729 scaled ground_truth')
        plot_df(time2, p_pred[:,0], 'tab:green', title= 'metric: '+i+' 0727-0729 model prediction')
        print('score(mean_absoluted_error):',score2)
        
        plot_df(time2, origin_pred_prediction,'tab:blue', title= 'metric: '+i+' 0727-0729 model original ground truth ')
        plot_df(time2, o_p_pred[:,0],'tab:green', title= 'metric: '+i+' 0727-0729 model inverse prediction')
        print('score(mean_absoluted_error):',score3)
        plt.show()
#         return p_pred, o_p_pred


# In[9]:


for i in columns:
    if version == 1:
        train, test, pred = extract_each_column(i)
        x_train, y_train = data_split(train, window_size)
        x_test, y_test = data_split(test, window_size)
        pred_history, pred_prediction = data_split(pred, window_size)
        # no use
        origin_pred_history, origin_pred_prediction = 0, 0
        # reshape to fit into model
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test= np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        pred_history = np.reshape(pred_history, (pred_history.shape[0],pred_history.shape[1],1))
        LSTM_model(x_train,y_train,x_test,y_test,pred_history,pred_prediction,origin_pred_history,origin_pred_prediction,model_type,version)
#         print(np.unique(p_pred))
    
    else:
        train, test, origin_test, origin_pred, pred = extract_each_column_v2(i)
        # scaled
        x_train, y_train = data_split(train, window_size)
        x_test, y_test = data_split(test, window_size)
        pred_history, pred_prediction = data_split(pred, window_size)
        # origin
        origin_pred_history, origin_pred_prediction = data_split(origin_pred, window_size)
        
        # reshape to fit into model
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test= np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        pred_history = np.reshape(pred_history, (pred_history.shape[0],pred_history.shape[1],1))
        origin_pred_history = np.reshape(origin_pred_history, (origin_pred_history.shape[0],origin_pred_history.shape[1],1))
        
        LSTM_model(x_train,y_train,x_test,y_test,pred_history,pred_prediction,origin_pred_history,origin_pred_prediction,model_type,version)
#         print(np.unique(p_pred))
#         print(np.unique(o_p_pred))

