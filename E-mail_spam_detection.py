#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all the libraries


# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve


# In[ ]:


#Adding the data in variable" data " using panadas library


# In[35]:


data = pd.read_csv('C:/Users/astit/Downloads/spam.csv', encoding='latin-1')


# In[36]:


data


# In[ ]:


# checking whether any line isn't null


# In[37]:


data.isnull().sum()


# In[ ]:


#taking only two columns from dataset and renaming v1 and v2 as label and message


# In[38]:


data = data[['v1', 'v2']]
data.columns = ['label', 'message']


# In[ ]:


#displaying the top 20 data from dataset


# In[39]:


data.head(20)


# In[ ]:


#copying the message column and to label column to variable x and y


# In[40]:


X = data["message"]
y = data["label"]


# In[ ]:


#converting the collection of raw documents into a matrix 


# In[41]:


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(X)


# In[ ]:


# split data into training and testing sets


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# applying multinomial naive baise


# In[43]:


clf = MultinomialNB()
clf.fit(X_train, y_train)


# In[44]:


y_pred = clf.predict(X_test)


# In[ ]:


#displaying the prediction


# In[49]:


y_pred


# In[ ]:


#checking the accuracy with original data and predicted data


# In[45]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# In[46]:


print(classification_report(y_test, y_pred))


# In[ ]:


# confusion Matrix


# In[47]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# In[48]:


labels = ['ham', 'spam']

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




