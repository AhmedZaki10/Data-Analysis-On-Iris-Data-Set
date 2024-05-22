#!/usr/bin/env python
# coding: utf-8

# In[748]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics.pairwise import *


# In[749]:


df = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\Iris\\data.csv')
pd.DataFrame(df)


# In[750]:


print(type(df))


# In[751]:


df.index


# In[752]:


df.columns


# In[753]:


df.info()


# In[754]:


df.shape


# In[755]:


df.head(10)


# In[756]:


df.tail(10)


# In[757]:


df['SepalLengthCm'].describe()


# In[758]:


df['SepalWidthCm'].describe()


# In[759]:


df['PetalLengthCm'].describe()


# In[760]:


df['PetalWidthCm'].describe()


# In[761]:


df.describe()


# In[762]:


df.mode()


# In[763]:


df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr()


# In[764]:


plt.scatter(x=df['SepalLengthCm'], y=df['PetalLengthCm'])
plt.title("Scatter Plot of Two Positive Correlated Terms")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[765]:


plt.scatter(x=df['SepalLengthCm'], y=df['PetalWidthCm'])
plt.title("Scatter Plot of Two Positive Correlated Terms")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[766]:


plt.scatter(x=df['PetalLengthCm'], y=df['PetalWidthCm'])
plt.title("Scatter Plot of Two Positive Correlated Terms")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[767]:


df.count()


# In[768]:


df.clip()


# In[769]:


df.round()


# In[770]:


df.rank()


# In[771]:


df['Class'].value_counts()


# In[772]:


df.isnull().any()


# In[773]:


df.drop_duplicates(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], inplace = True)


# In[774]:


df.head(20)


# In[775]:


df.shape


# In[776]:


df.plot.bar()


# In[777]:


df.plot.box()


# In[778]:


df.plot.hist()


# In[779]:


df.plot()


# In[780]:


df.plot.area()


# In[781]:


df.plot.barh()


# In[782]:


df.plot.density()


# In[783]:


df.plot.hexbin(x='SepalLengthCm',y='SepalWidthCm')


# In[784]:


df.plot.hexbin(x='PetalLengthCm',y='PetalWidthCm')


# In[789]:


df.plot.scatter(x='SepalLengthCm',y='SepalWidthCm')


# In[790]:


df.plot.scatter(x='PetalLengthCm',y='PetalWidthCm')


# In[791]:


df.hist(edgecolor='black', linewidth=1.2)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[792]:


df.hist()


# In[793]:


df[['SepalLengthCm','Class']].groupby('Class' , as_index = False).mean()


# In[794]:


df[['SepalWidthCm','Class']].groupby('Class' , as_index = False).mean()


# In[795]:


df[['PetalLengthCm','Class']].groupby('Class' , as_index = False).mean()


# In[796]:


df[['PetalWidthCm','Class']].groupby('Class' , as_index = False).mean()


# In[797]:


df[['SepalLengthCm','Class']].groupby('SepalLengthCm' , as_index = False).count()


# In[798]:


df[['SepalWidthCm','Class']].groupby('SepalWidthCm' , as_index = False).count()


# In[799]:


df[['PetalLengthCm','Class']].groupby('PetalLengthCm' , as_index = False).count()


# In[800]:


df[['PetalWidthCm','Class']].groupby('PetalWidthCm' , as_index = False).count()


# In[801]:


df[['SepalLengthCm','Class']].groupby('Class' , as_index = False).median()


# In[802]:


df[['SepalWidthCm','Class']].groupby('Class' , as_index = False).median()


# In[803]:


df[['PetalLengthCm','Class']].groupby('Class' , as_index = False).median()


# In[804]:


df[['PetalWidthCm','Class']].groupby('Class' , as_index = False).median()


# In[805]:


df[['SepalLengthCm','Class']].groupby('Class' , as_index = False).sum()


# In[806]:


df[['SepalWidthCm','Class']].groupby('Class' , as_index = False).sum()


# In[807]:


df[['PetalLengthCm','Class']].groupby('Class' , as_index = False).sum()


# In[808]:


df[['PetalWidthCm','Class']].groupby('Class' , as_index = False).sum()


# In[809]:


has_high_sepal_length = df['SepalLengthCm'] >= 5.0
has_high_sepal_width = df['SepalWidthCm'] >= 3.0
has_high_petal_length = df['PetalLengthCm'] >= 4.0
has_high_petal_width = df['PetalWidthCm'] >= 1.5
df_new = df[has_high_sepal_length & has_high_sepal_width & has_high_petal_length & has_high_petal_width]
df_new


# In[810]:


df_new.plot(kind="bar",figsize=(15,10))


# In[811]:


df


# In[812]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
euclidean = pairwise_distances(df, metric='euclidean')
print(euclidean)


# In[813]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
manhattan = pairwise_distances(df, metric='manhattan')
print(manhattan)


# In[814]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
squared_euclidean = pairwise_distances(df, metric='euclidean' , squared = True)
print(squared_euclidean)


# In[815]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
hamming_distances = pairwise_distances(df, metric='hamming')
print(hamming_distances)


# In[816]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
city_block_distances = pairwise_distances(df, metric='cityblock')
print(city_block_distances)


# In[817]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
p_value = int(input('Enter an integer: ')) 
minkowski_distances = pairwise_distances(df, metric='minkowski', p=p_value)
print(minkowski_distances)


# In[818]:


columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[columns]
cosine_similarities = 1 - cosine_similarity(df)
print(cosine_similarities)


# In[820]:


# Convert the matrices to DataFrames
euclidean_df = pd.DataFrame(euclidean)
manhattan_df = pd.DataFrame(manhattan)
squared_euclidean_df = pd.DataFrame(squared_euclidean)
hamming_df = pd.DataFrame(hamming_distances)
cityblock_df = pd.DataFrame(city_block_distances)
mainkowski_df = pd.DataFrame(minkowski_distances)
cosine_df = pd.DataFrame(cosine_similarities)

# Combine them into one DataFrame
all_distances = pd.concat([euclidean_df, manhattan_df,squared_euclidean_df, hamming_df, cityblock_df, mainkowski_df, cosine_df], axis=1)

# Save all dissimilarity matrices to a single CSV file
all_distances.to_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\Dissimilarity Matrix\\dissmilarity_matrix.csv', index=False, header=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




