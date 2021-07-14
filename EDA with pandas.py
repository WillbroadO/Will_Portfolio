#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)


# In[24]:


df = pd.read_excel("Telco Churn dataset.xlsx")
df.head()


# In[25]:


# Let’s have a look at data dimensionality, feature names, and feature types.
print(df.shape)


# From the output, we can see that the table contains 3333 rows and 20 columns.
# 
# Now let's try printing out column names using columns:

# In[26]:


print(df.columns)


# We can use the info() method to output some general information about the dataframe

# In[27]:


df.info()


# We can change the column type with the astype method. 
# Let's apply this method to the Churn feature to convert it into int64:

# The describe method shows basic statistical characteristics of each numerical feature (int64 and float64 types)

# In[28]:


df.describe()


# In order to see statistics on non-numerical features, one has to explicitly indicate data types of interest in the include parameter.

# In[29]:


df.describe(include=["object", "bool"])


# In[41]:


# For categorical (type object) and boolean (type bool) features we can use the value_counts method. Let's have a look at the distribution of Churn:


# In[43]:


df["Churn"].value_counts()


# In[31]:


df["Churn"].value_counts(normalize=True) # we pass normalize = True to calculate the fractions


# Sorting
# A DataFrame can be sorted by the value of one of the variables (i.e columns). For example, we can sort by Total Call (use ascending=False to sort in descending order):

# In[32]:


df.sort_values(by="TotalCall", ascending=False).head()


# In[33]:


df.sort_values(by=["TotalCall","Churn"], ascending=[True, False]).head() # sortinb by multiple columns


# In[39]:


from sklearn.preprocessing import LabelEncoder
df['Churn_encoded'] = LabelEncoder().fit_transform(df['Churn'])
df[['Churn', 'Churn_encoded']] # special syntax to get just these two columns, we have encoded yes and no to 1 and 0


# In[38]:


df.head()


# # Indexing and retrieving data

# In[45]:


# A DataFrame can be indexed in a few different ways. what is the proportion of churned users in our dataframe?


# In[44]:


df["Churn_encoded"].mean()


# In[48]:


# Average values of numerical features for churned users?


# In[47]:


df[df["Churn_encoded"] == 1].mean()


# In[49]:


df[df["Churn_encoded"] == 1]["TotalDayMinutes"].mean() # average time churned users spend on the phone during daytime?


# In[50]:


df[df["Churn_encoded"] == 1]["TotalDayMinutes"].max() # max time of a churned user


# In[51]:


df[df["Churn_encoded"] == 1]["TotalDayMinutes"].min() # max time of a churned user


# In[53]:


# What is the maximum length of international calls among loyal users (Churn == 0) who do not have an international plan?


# In[60]:


df[(df["Churn"] == 0) & (df["InternationalPlan"] == "No")]["TotalIntlMinutes"].max()


# DataFrames can be indexed by column name (label) or row name (index) or by the serial number of a row. The loc method is used for indexing by name, while iloc() is used for indexing by number.

# In[61]:


df.loc[0:5, "customerID":"OnlineBackup"]


# In[62]:


df.iloc[0:5, 0:3]


# In[ ]:


# If we need the first or the last line of the data frame, we can use the df[:1] or df[-1:] construct:


# In[63]:


df[:1]


# Applying Functions to Cells, Columns and Rows
# To apply functions to each column, use apply():

# In[66]:


# Lambda functions are very convenient in such scenarios. For example, if we need to select all InternetService starting with F, we can do it like this:


# In[68]:


df[df["InternetService"].apply(lambda InternetService: InternetService [0] == "F")].head()


# The map method can be used to replace values in a column by passing a dictionary of the form {old_value: new_value} as its argument:

# In[69]:


d = {"No": False, "Yes": True}
df["InternationalPlan"] = df["InternationalPlan"].map(d)
df.head()


# In[70]:


df["InternationalPlan"]


# In[71]:


df = df.replace({"VoiceMailPlan": d}) # replace method can be used in place of map
df.head()


# # Grouping

# In[73]:


# In pandas groupby can be by below method


# In[74]:


# df.groupby(by=grouping_columns)[columns_to_show].function()


# Here is an example where we group the data according to the values of the Churn variable and display statistics of three columns in each group:

# In[76]:


columns_to_show = ["TotalDayMinutes", "CustomerServiceCalls","TotalCall"]

df.groupby(["Churn"])[columns_to_show].describe(percentiles=[])


# In[77]:


# Conversely lets pass the list of functions to agg


# In[78]:


columns_to_show = ["TotalDayMinutes", "CustomerServiceCalls","TotalCall"]
df.groupby(["Churn"])[columns_to_show].agg([np.mean, np.std, np.min, np.max])


# # Summary tables

# Suppose we want to see how the observations in our sample are distributed in the context of two variables - Churn and International plan. To do so, we can build a contingency table using the crosstab method:

# In[79]:


pd.crosstab(df["Churn"], df["InternationalPlan"])


# In[80]:


pd.crosstab(df["Churn"], df["VoiceMailPlan"], normalize=True)


# This will resemble pivot tables to those familiar with Excel. And, of course, pivot tables are implemented in Pandas: the pivot_table method takes the following parameters:
# 
# values – a list of variables to calculate statistics for,
# index – a list of variables to group data by,
# aggfunc – what statistics we need to calculate for groups, ex. sum, mean, maximum, minimum or something else.

# In[83]:


df.pivot_table(
    ["NumbervMailMessages","TotalDayMinutes","TotalDayCalls"],
    ["PaymentMethod"],
    aggfunc="mean",
)


# # DataFrame transformations

# Like many other things in Pandas, adding columns to a DataFrame is doable in many ways.
# 
# For example, if we want to calculate the total number of calls for all users, let's create the total_calls Series and paste it into the DataFrame:

# In[84]:


total_calls = (
    df["TotalDayCalls"]
    + df["TotalEveCalls"]
    + df["TotalNightCalls"]
    + df["TotalIntlCalls"]
)
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
# loc parameter is the number of columns after which to insert the Series object
# we set it to len(df.columns) to paste it at the very end of the dataframe
df.head()


# In[86]:


# It is possible to add a column more easily without creating an intermediate Series instance:


# In[87]:


df["Total Minutes"] = (
    df["TotalDayMinutes"]
    + df["TotalEveMinutes"]
    + df["TotalNightMinutes"]
    + df["TotalIntlMinutes"]
)
df.head()


# To delete columns or rows, use the drop method, passing the required indexes and the axis parameter (1 if you delete columns, and nothing or 0 if you delete rows). The inplace argument tells whether to change the original DataFrame. With inplace=False, the drop method doesn't change the existing DataFrame and returns a new one with dropped rows or columns. With inplace=True, it alters the DataFrame.

# In[98]:


# get rid of just created columns
# df.drop(["Total calls"], axis=1, inplace=True)


# In[92]:


# and here’s how you can delete rows
df.drop([1, 2]).head()

