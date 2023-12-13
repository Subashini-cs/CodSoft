#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Importing dataset

# In[2]:


url = 'C:/Users/subas/Downloads/archive (2).zip'
df = pd.read_csv(url, encoding='latin-1')
df = df[['v1', 'v2']] 
df


# In[3]:


df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
df


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)


# In[5]:


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
vectorizer


# In[6]:


classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)


# In[7]:


predictions = classifier.predict(X_test_vec)
predictions


# In[8]:


accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


# # Accuracy of dataset

# In[11]:


print(f'Accuracy: {accuracy}')


# # Confusion matrix

# In[12]:


print('\nConfusion Matrix:')
print(conf_matrix)


# # Classification report

# In[9]:


print('\nClassification Report:')
print(classification_rep)

