#!/usr/bin/env python
# coding: utf-8

# # Lego Data Exploration
# 
# ## Price prediction models

# [Data exploration](index.html)
# 
# [Price prediction models (current page)](regression.html)

# ### Regression using Year only
# 
# We'll first try linear reression.

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 216
plt.style.use('fivethirtyeight')

data = pd.read_pickle('sets.pkl')
year_list = list(range(1978, 2021))
inflation_data = pd.read_csv('inflation_data.csv')

def get_ppp_list_all(): # ppp = price per piece
    ppp_list = []
    for year in year_list:
        year_data = data[data['Year'] == year]
        total_price = year_data['USRetailPrice'].sum(axis='index')
        total_pieces = year_data['Pieces'].sum(axis='index')
        inflation_price = inflation_data[inflation_data['year'] == year]['amount'].iat[0]
        ppp = total_price / total_pieces * inflation_price
        if ppp == 0:
            ppp = np.nan
        ppp_list.append(ppp)        
    return ppp_list

ppp_list = get_ppp_list_all()


# In[22]:


x = np.array(year_list).reshape((-1,1))
y = ppp_list
model = LinearRegression().fit(x, y)
y_pred = model.predict(x)

r_sq = model.score(x,y)
r_sq


# The r^2 value is 0.8, which indicates a decent fit to the data. However, the price has been dropping in recent years, and there are peaks and troughs throughout the data. While we can say that the price generally increases in the long run, this simple model would fall short in making short-term predictions.

# In[37]:


plt.figure(figsize=(10,7)) 
plt.xlabel('Year')
plt.ylabel('Price per piece [$USD]')
plt.title('Year vs. Price (US Retail) per Piece')
plt.plot(year_list, ppp_list, label='Average')
plt.plot(x, y_pred, linewidth=4)


# ### Regression using Year, Pieces, Minifigs, Theme

# In[24]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# In[25]:


data['Theme'].isnull().values.any()


# In[26]:


temp = data[['Year', 'Minifigs', 'Pieces','USRetailPrice']]
temp2 = pd.get_dummies(data['Theme']) # Turn Theme into a dummy variable
data_new = pd.concat([temp, temp2], axis=1)
# Drop NaN's and 0's for price
data_new = data_new[data_new['USRetailPrice'].notna() & (data_new['USRetailPrice'] != 0)] 
data_new = data_new.replace(np.nan, 0)


# In[27]:


# Separate into training and testing sets
train_dataset = data_new.sample(frac=0.8,random_state=0)
test_dataset = data_new.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop('USRetailPrice')
train_stats = train_stats.transpose()
train_stats


# The features have significantly different ranges, so we should normalize them to avoid dependence on our choice of unit. 
# 
# 

# In[28]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[29]:


normed_train_data = train_dataset
normed_test_data = test_dataset


# In[30]:


train_labels = train_dataset.pop('USRetailPrice')
test_labels = test_dataset.pop('USRetailPrice')


# In[31]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
        optimizer='adam',
        metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()


# In[41]:


EPOCHS = 500

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
#     callbacks=[tfdocs.modeling.EpochDots()]
)


# In[42]:


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plt.figure(figsize=(10,7)) 
plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 1000])
plt.ylabel('MSE [USRetailPrice]')


# In[43]:


plt.figure(figsize=(10,7)) 
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 30])
plt.ylabel('MAE [USRetailPrice]')


# In[44]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
mae


# Issues:
# - MAE is \\$8.30 after one run, which is pretty high
# - Normalizing doesn't seem to work (?)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




