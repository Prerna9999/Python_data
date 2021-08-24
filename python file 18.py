#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv('Hotel_reviews.csv')


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.columns


# In[8]:


data.head()


# In[15]:


def ratings(x):
    if x==5:
        return 'outstanding'
    elif x==4:
        return 'excellent'
    elif x==3:
        return 'Average'
    elif x==2 and x == 1:
        return 'Good'
    else:
        return 'poor'
        
data['marking']=data['Rating'].apply(ratings)


# In[16]:


data.head()


# In[19]:


data = data.drop(['ratings'],axis=1)


# In[20]:


data.head()


# In[40]:


#punctuation = string.punctuation
#data['punctuation_count'] = data['Review'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 


# In[41]:


#data.head()


# In[42]:


#data[['Rating']].describe()


# In[31]:


import pandas as pd
get_ipython().system(' pip install textblob')

from textblob import TextBlob


# In[32]:


data['length'] = data['Review'].apply(len)


# In[33]:


data.head()


# In[35]:


def get_polarity(text):
    textblob = TextBlob(str(text.encode('utf-8')))
    pol = textblob.sentiment.polarity
    return pol


data['polarity'] = data['Review'].apply(get_polarity)


# In[37]:


def get_subjectivity(text):
    textblob = TextBlob(str(text.encode('utf-8')))
    subj = textblob.sentiment.subjectivity
    return subj

data['subjectivity'] = data['Review'].apply(get_subjectivity)
data.head()


# In[43]:


data[['length','polarity','subjectivity']].describe()



# In[ ]:




