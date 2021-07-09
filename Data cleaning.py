#!/usr/bin/env python
# coding: utf-8

# # Project 1: Data Cleaning

# * Handling missing values
# * Data scaling and normalization
# * Cleaning and parsing dates
# * Character encoding errors (no more messed up text fields!)
# * Fixing inconsistent data entry & spelling errors
# 
# 
# 
# Building permit data: A building permit is an official approval document issued by a governmental agency that allows you or your contractor to proceed with a construction or remodeling project on one's property. For more details go to https://www.thespruce.com/what-is-a-building-permit-1398344. Each city or county has its own office related to buildings, that can do multiple functions like issuing permits, inspecting buildings to enforce safety measures, modifying rules to accommodate needs of the growing population etc. For the city of San Francisco, permit issuing is taken care by www.sfdbi.org/

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


Building_P = pd.read_csv('Building_Permits.csv')

# set seed for reproducibility so anyone who runs the code get exact same output
np.random.seed(0) 

# sample the dataset
Building_P.sample (5) 


# In[4]:


# I get the number of missing data points per column
missing_values_count = Building_P.isnull().sum()

# I look at the # of missing points in the first ten columns
missing_values_count[0:10]


# In[5]:


# how many total missing values do we have?
total_cells = np.product(Building_P.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100


# In[6]:


# look at the # of missing points in the first ten columns
missing_values_count[0:10]


# In[7]:


# remove all the rows that contain a missing value
Building_P.dropna()


# In[8]:


Building_P.shape


# In[9]:


# remove all columns with at least one missing value
columns_with_na_dropped = Building_P.dropna(axis=1)
columns_with_na_dropped.head()


# In[10]:


columns_with_na_dropped.shape


# In[34]:


# just how much data did we lose?
print("Columns in original dataset: %d \n" % Building_P.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])


# In[ ]:


# We have lost 31 columns not feasible at al


# # Filling in missing values automatically

# In[17]:


# get a small subset of the Building Permit
subset_Building_P = Building_P.loc[:, 'Permit Number':'Record ID'].head()
subset_Building_P


# In[15]:


# replace all NA's with 0
subset_Building_P.fillna(0)


# # I could also be a bit more savvy and replace missing values with whatever value comes directly after it in the same column.
# (This makes a lot of sense for datasets where the observations have some sort of logical order to them.)
# 
# 

# In[16]:


# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
# subset_Building_P.fillna(method = 'bfill', axis=0).fillna(0)

