#!/usr/bin/env python
# coding: utf-8

# # Objective
# The purpose is to create a demonstable prototype that mines purchase data and predicts categories similar to the input.
# For elucidations sake we will divide the summary problem into sub problems.
# *  Problem 1 -Given item A predict item B which is most associated through purchase patterns.
# 

# # Packages required
# * Pandas for data frame
# * Numpy for arrays

# In[1]:


import pandas as pd
import numpy as np
from scipy import sparse


# # Problem 1

# # Import Data

# In[2]:


data = pd.read_csv('../data/FMCGSales.csv', names = ['BillId','ItemId','ItemName','Level1','Level2','Level3','Level4','Level5','Level6'] )


# In[3]:


#Dummy for pivot table
data['dummy'] = 1


# # Items Index

# In[4]:


names = data[['ItemName', 'ItemId']]
names = names.drop_duplicates(subset = 'ItemName')



# # Pivot Table

# In[5]:


matrix = data.pivot_table(values='dummy',index ='BillId', columns ='ItemId')


# In[6]:


matrix_dummy = matrix.copy().fillna(0)


# # Jaccardian
# We define a jaccardian as intersection over union. A skewed Jaccardian is intersection over set A

# ### method from http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html

# In[7]:


sparse_matrix = sparse.csc_matrix(matrix_dummy)


# In[8]:


def jaccard_similarities(mat):
    cols_sum = mat.getnnz(axis=0)
    ab = mat.T * mat

    # for rows
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    # for columns
    bb = cols_sum[ab.indices]

    similarities = ab.copy()
    similarities.data /= (aa + bb - ab.data)

    return similarities


# In[9]:


jaccard_similarities =  jaccard_similarities(sparse_matrix)


# In[10]:


jaccardian = pd.DataFrame(jaccard_similarities.toarray(), index = matrix.columns,columns = matrix.columns)


# # Final Function

# In[11]:


# Final Function
# Takes title, userId and returns to 25 sorted movie names on est
def reco(idx):

    similar_itemids = jaccardian.loc[idx].to_frame(name=None)
    topItemIds = similar_itemids.sort_values(by =[idx],ascending = False).head(26)
    topItemNames = topItemIds.merge(names, how ='left', on = 'ItemId' )

    
    return topItemNames.head(10)


# In[12]:


print(reco(110828))


# In[13]:


reco(203289)


# In[14]:


reco(220862)

