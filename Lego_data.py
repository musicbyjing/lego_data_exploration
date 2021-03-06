#!/usr/bin/env python
# coding: utf-8

# # Lego Data
# ## Dataset exploration

# [Dataset exploration (current page)](index.html)
# 
# [Price prediction models](regression.html)

# As a longtime Lego fan and peruser of the [Brickset](https://brickset.com/) fansite, I decided to see what interesting trends I could find in their comprehensive database. Over time, my interest in the monetary aspect of Lego has grown: How much do sets cost? Do they cost more nowadays compared to before? Does a given set cost more or less than average? What are the relevant variables for that calculation? (The inescapable question of why Lego is so expensive in general will be ignored, for my sanity.) After exploring these questions, I will build a model to predict Lego prices based on their features.
# 
# Lego also has a thriving secondary market, where certain retired sets [rack in absurd returns](https://www.catawiki.com/stories/715-top-10-most-expensive-lego-sets). I will then bring in data for current secondary-market prices to build a model to predict which *current* sets might make money in the future.
# 

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 216
plt.style.use('fivethirtyeight')


# ## Preliminary questions
# - Are longstanding current themes more likely to have a greater number of set releases per year?
# - Has the price per piece increased over time? (Pick a specific theme, e.g. Star Wars)
#     - Does this align with inflation? (Pick a specific currency)
# - Are some themes more expensive than others? (Pick a specific year range)

# In[7]:


curr_themes = ['Architecture', 'Brick Sketches', 'BrickHeadz', 'City', 'Classic',                'Collectable Minifigures', 'Creator', 'Creator Expert', 'DC Comics Super Heroes'                 'Disney', 'DOTS', 'Duplo', 'Education', 'Friends', 'Harry Potter',                'Hidden Side', 'Ideas', 'Jurassic World', 'LEGO Art', 'Marvel Super Heroes',                'Mindstorms', 'Minecraft', 'Minions: The Rise of Gru', 'Monkie Kid', 'Ninjago'                'Overwatch', 'Powered Up', 'Speed Champions', 'Star Wars', 'Super Heroes',                'Super Mario', 'Technic ', 'Trolls World Tour']


data = pd.read_csv('allsets.csv')
print('Table has shape:', data.shape)
data.head()


# ## Visualizing numerical values: Year, Minifigs, Pieces, USRetailPrice

# In[8]:


fig, axes = plt.subplots(nrows = 2, ncols = 2)
num_features = ['Year', 'Minifigs', 'Pieces', 'USRetailPrice'] # features of interest
axes = axes.ravel()
for i, ax in enumerate(axes):
    if i == 0:
        ax.hist(data[num_features[i]].dropna(), bins=40)
    else:
        ax.hist(data[num_features[i]].dropna(), bins=40, log=True)
    ax.set_xlabel(num_features[i], fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
fig.set_size_inches(20, 10)


# A large amount of entries have no pieces (`NaN` or `0`, from looking at the CSV). Either they are not sets (e.g. books, pens, video games, etc.) or the data is incomplete. Regardless, we will discard them.

# In[9]:


data = data[~(pd.isna(data['Pieces']) | (data['Pieces']==0))]
print('Table has shape:', data.shape)


# ## Visualizing categorical values: ThemeGroup

# In[10]:


ax = data.groupby('ThemeGroup').size().sort_values().plot(kind='barh', figsize=(10,7))
ax.set_xlabel('Count')


# ## Are the numerical values correlated?

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.pairplot(data[["Year", "Minifigs", "Pieces", "USRetailPrice"]], diag_kind="kde")


# The most striking correlation is that the number of pieces is strongly correlated with price. In other words, larger sets tend to be more expensive.
# 
# More interestingly, there is a trend of sets getting bigger and more expensive over the years. Let's use the data to confirm this hypothesis.

# ## Has price per piece increased over time?
# 
# First, we'll look at the whole dataset. 
# 
# Since the first Lego themes were introduced in 1978, we must consider inflation--\\$1 in 1978 is worth more than \\$1 in 2020. Luckily, I found a [CSV file](https://www.in2013dollars.com/us/inflation/1978?amount=1) that lets us account for this.

# In[12]:


year_list = list(range(1978, 2021)) # The first Lego themes were introduced in 1978
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

plt.figure(figsize=(10,7)) 
plt.xlabel('Year')
plt.ylabel('Price per piece [$USD]')
plt.title('Year vs. Price (US Retail) per Piece')
plt.plot(year_list, ppp_list, label='Average', linewidth=4)


# The overall trend seems to be increasing. 
# 
# But some theme groups, such as Licensed, have a reputation in the Lego community for being more expensive than others. Let's see if this is true.
# 

# In[13]:


# theme_groups = ['Licensed', 'Miscellaneous', 'Modern day', 'Pre-school', 'Action/Adventure', \
#                 'Basic', 'Girls', 'Model making', 'Technical', 'Constraction', 'Historical', \
#                 'Vintage themes', 'Educational', 'Racing', 'Junior']

def plot_ppp_list(theme): 
    theme_data = data[data['ThemeGroup'] == theme]
    ppp_list = []
    for year in year_list:
        year_data = theme_data[theme_data['Year'] == year] # pandas df version of filter
        total_price = year_data['USRetailPrice'].sum(axis='index') # sum down the column
        total_pieces = year_data['Pieces'].sum(axis='index')
        inflation_price = inflation_data[inflation_data['year'] == year]['amount'].iat[0]
        if total_price == 0 or total_pieces == 0:
            ppp = np.nan
        else:
            ppp = total_price / total_pieces * inflation_price
        ppp_list.append(ppp)        
    plt.plot(year_list, ppp_list, label=theme, linestyle='--', linewidth=2)

plt.figure(figsize=(10,7)) 
plt.xlabel('Year')
plt.ylabel('Price per piece [$USD]')
plt.title('Year vs. Price (US Retail) per Piece')
plt.plot(year_list, ppp_list, label='Average', linewidth=4)
plot_ppp_list("Licensed")
plt.legend()


# In 1999, the year of their introduction, Licensed sets start off significantly cheaper than average, but grow more expensive in the late 2000's. After 2010 they seem to be par for the course.
# 
# Let's plot a few more theme groups to compare.

# In[14]:


plt.figure(figsize=(10,7)) 
plt.xlabel('Year')
plt.ylabel('Price per piece [$USD]')
plt.title('Year vs. Price (US Retail) per Piece')
plt.plot(year_list, ppp_list, label='Average')
plot_ppp_list("Licensed")
plot_ppp_list("Action/Adventure")
plot_ppp_list("Miscellaneous")
plot_ppp_list("Modern day")
plt.legend()


# What's responsible for the spike in Miscellaneous at 1997?

# In[15]:


temp = data[(data['ThemeGroup'] == "Miscellaneous") & (data['Year'] == 1997)]
temp[['SetID', 'Theme', 'Year', 'Name', 'Pieces', 'USRetailPrice']]


# It seems Service Packs are driving up the yearly average. For example, set 2742 (SetID) is a one piece battery that sold for \\$5. 
# 
# It isn't really a "set", per se, so given this and similar occurrences, it might be best to exclude the Miscellaneous theme entirely.

# In[16]:


data = data[~(data['ThemeGroup'] == 'Miscellaneous')]
print('Table has shape:', data.shape)


# In[17]:


data.to_pickle('sets.pkl')

