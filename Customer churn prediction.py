#!/usr/bin/env python
# coding: utf-8

# # Installing 'xgboost' library

# In[1]:


get_ipython().system('pip install xgboost')


# # Importing libraries

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Importing the dataset

# In[3]:


data_set = pd.read_csv("Churn_Modelling.csv")
data_set


# # Dropping columns

# In[4]:


data_set = pd.get_dummies(data_set, columns=['Geography', 'Gender'], drop_first = True)
data_set


# # spliting dependent and independent values

# In[5]:


x = data_set.drop(columns = ['Exited', 'RowNumber', 'CustomerId', 'Surname'])
y = data_set['Exited']
x


# In[6]:


y


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)


#  # Building the model

# In[8]:


model = XGBClassifier(n_estimators=10)
model.fit(x_train, y_train)


#  # Prediction of model

# In[14]:


y_pred = model.predict(x_test)
new_dataset=np.array([[589,39,1,0.00,1,1,1,129456.76,0,0,0]])
y_pred1 = model.predict(new_dataset)
y_pred1


# # Accuracy of the model

# In[15]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# # Confusion matrix

# In[11]:


confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)


# # Classification report

# In[16]:


classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

