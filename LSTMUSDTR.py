#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import itertools
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle

import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')


# In[4]:


#dolar kur datasini yukluyorum & I load the dollar currency file
data = pd.read_excel('usd_alis.xlsx', date_parser=[0])

# İlk 5 Satır & And let's see first 5 row
data.head()


# In[5]:



#Datetime Haline Getirilmesi & Make it datetime shape
data['tarih'] = pd.to_datetime(data.tarih, format='%d-%m-%Y')
 
#İndex'e Alınması & Put it in index
data.index = data.tarih


# In[6]:



#Burada interpolate fonksiyonunu kullanıyoruz. Haftaya bakarak lineer olarak dolduruyor. & We use interpolate function it fiils week by week

data['usd_alis'].interpolate(method='linear', inplace=True)



# In[7]:



fig = plt.figure(figsize=(15,8))
data.usd_alis.plot(label='usd alış')
plt.legend(loc='best')
plt.title('Daily Exchange Rates, Buy', fontsize=14)
plt.show()


# In[8]:


values = data['usd_alis'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)
 

dataset[0:5]


# In[9]:


# %60 Train % 40 Test
TRAIN_SIZE = 0.60

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Gün Sayıları (training set, test set): " + str((len(train), len(test))))


# In[10]:


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))


# In[11]:


def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    # Modelin tek layerlı şekilde kurulacak. & Built in a single layer
    model.add(LSTM(100, 
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam")
 
    model.fit(train_X, 
              train_Y, 
              epochs = 30, 
              batch_size = 1, 
              verbose = 1)
    
    return(model)
 
# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)


# In[12]:


def predict_and_score(model, X, Y):
    # Tahminleri 0-1 ile scale edilmiş halinden geri çeviriyoruz. & Scale between 1-0 on Predictive values and retured back
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    # Rmse değerlerini ölçüyoruz. & Measure RMSE Values
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)
 
rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
 
print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)


# In[13]:


# Tahminletip ekliyoruz. & We predict and add
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Testleri tahminletiyoruz. & Pradiction on test
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict

# Tahminin Plot'unu oluşturalım. & Let's make a Plot
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Days")
plt.ylabel("Exchange Rates")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()


# In[ ]:




