#!/usr/bin/env python
# coding: utf-8

# # Building a binary classification machine learning model to predict if the a customer will leave after six months

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Loading data

# In[3]:


customer_data = pd.read_csv("Churn_Modelling.csv")
customer_data.head()


# ## Feature selection

# In[4]:


columns = customer_data.columns.values.tolist()
print(columns)


# In[5]:


dataset = customer_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) # dropping columns with no influence to the dependent variable


# In[6]:


dataset.head()


# ## Converting Categorical Columns to Numeric Columns

# Machine learning algorithms work best with numerical data. However, in our dataset, we have two categorical columns: Geography and Gender. These two columns contain data in textual format; we need to convert them to numeric columns.

# In[7]:


dataset =  dataset.drop(['Geography', 'Gender'], axis=1) # isoloate the two columns
dataset


# One way to convert categorical columns to numeric columns is to replace each category with a number. For instance, in the Gender column, female can be replaced with 0 and male with 1, or vice versa. This works for columns with only two categories.

# Another better way to convert such categorical columns to numeric columns is by using one-hot encoding. In this process, we take our categories (France, Germany, Spain) and represent them with columns. In each column, we use a 1 to designate that the category exists for the current row, and a 0 otherwise.

# In[8]:


Geography = pd.get_dummies(customer_data.Geography).iloc[:,1:]
Gender = pd.get_dummies(customer_data.Gender).iloc[:,1:]


# The get_dummies method of the pandas library converts categorical columns to numeric columns. Then, .iloc[:,1:] ignores the first column and returns the rest of the columns (Germany and Spain). As noted above, this is because we can always represent "n" categories with "n - 1" columns.
# 
# Now if you open the Geography and customer_data data frames in the Variable Explorer pane, you should see something like this:

# In[9]:


dataset = pd.concat([dataset,Geography,Gender], axis=1) # adding back geography and gender columns back
dataset.head()


# ## Data Preprocessing

# In[14]:


X =  dataset.drop(['Exited'], axis=1) # Here, X is our feature set; it contains all the columns except the one that we have to predict (Exited). The label set, y, contains only the Exited column.
y = dataset['Exited']


# So we can later evaluate the performance of our machine learning model, let's also divide the data into a training and test set. The training set contains the data that will be used to train our machine learning model. The test set will be used to evaluate how good our model is. We'll use 20% of the data for the test set and the remaining 80% for the training set (specified with the test_size argument):

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Machine Learning Algorithm training

# Now, we'll use a machine learning algorithm that will identify patterns or trends in the training data. This step is known as algorithm training. We'll feed the features and correct output to the algorithm; based on that data, the algorithm will learn to find associations between the features and outputs. After training the algorithm, you'll be able to use it to make predictions on new data.
# 
# There are several machine learning algorithms that can be used to make such predictions. However, we'll use the RANDOM FOREST ALGORITHM, since it's simple and one of the most powerful algorithms for classification problems.
# 
# To train this algorithm, we call the fit method and pass in the feature set (X) and the corresponding label set (y). You can then use the predict method to make predictions on the test set. Look at the following script:

# In[16]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)


# ## Machine learning Algorithm evaluation

# Now that the algorithm has been trained, it's time to see how well it performs. For evaluating the performance of a classification algorithm, the most commonly used metrics are the F1 MEASURE, PRECISION, RECALL, AND ACCURACY. In Python's scikit-learn library, you can use built-in functions to find all of these values. Execute the following script:

# In[17]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))


# The results indicate an accuracy of 86.35%, which means that our algorithm successfully predicts customer churn 86.35% of the time. That's pretty impressive for a first attempt!

# ## Feature Evaluation

# As a final step, let's see which features play the most important role in the identification of customer churn. Luckily, RandomForestClassifier contains an attribute named feature_importance that contains information about the most important features for a given classification.
# 
# The following code creates a bar plot of the top 10 features for predicting customer churn:

# In[18]:



feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


# Based on this data, we can see that age has the highest impact on customer churn, followed by a customer's estimated salary and account balance

# Conclusion
# 
# Customer churn prediction is crucial to the long-term financial stability of a company. I have successfully created a machine learning model that's able to predict customer churn with an accuracy of 86.35%.
