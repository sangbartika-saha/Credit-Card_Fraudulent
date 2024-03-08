#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


get_ipython().system('pip install SVC')


# In[3]:


train_data = pd.read_csv("E:/projects/Internship/Credit card/Dataset/fraudTrain.csv")


# In[4]:


train_data.info()


# In[5]:


train_data.describe()


# In[6]:


train_data.dtypes


# In[7]:


train_data.columns


# In[8]:


train_data["trans_date_trans_time"] = pd.to_datetime(train_data["trans_date_trans_time"])
train_data["dob"] = pd.to_datetime(train_data["dob"])
train_data


# In[9]:


train_data


# In[10]:


train_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
train_data


# In[11]:


#Drop all rows that contain missing values 
train_data.dropna().reset_index(drop=True)
train_data


# In[12]:


train_data


# In[13]:


encoder = LabelEncoder()
train_data["merchant"] = encoder.fit_transform(train_data["merchant"])
train_data["category"] = encoder.fit_transform(train_data["category"])
train_data["gender"] = encoder.fit_transform(train_data["gender"])
train_data["job"] = encoder.fit_transform(train_data["job"])


# In[14]:


train_data


# # EDA

# In[16]:


exit_counts = train_data["is_fraud"].value_counts()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Subplot for the pie chart
plt.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
plt.title("Fraud Counts")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# # TRAIN MODEL

# In[17]:


X = train_data.drop(columns=["is_fraud"], inplace = False)
Y = train_data["is_fraud"]


# In[ ]:


model = SVC()
model.fit(X,Y)


# In[23]:


get_ipython().system('pip install SVM')


# In[ ]:


model.score(X, Y)


# In[44]:


train_data


# # Test model
# 

# In[46]:


test_data = pd.read_csv("E:/projects/Internship/Credit card/Dataset/fraudTest.csv")
test_data


# In[47]:


test_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
test_data


# In[48]:


encoder = LabelEncoder()
test_data["merchant"] = encoder.fit_transform(test_data["merchant"])
test_data["category"] = encoder.fit_transform(test_data["category"])
test_data["gender"] = encoder.fit_transform(test_data["gender"])
test_data["job"] = encoder.fit_transform(test_data["job"])


# In[49]:


test_data


# In[50]:


X_test = test_data.drop(columns=["is_fraud"], inplace = False)
Y_test = test_data["is_fraud"]


# In[51]:


y_pred = model.predict(X_test)
y_pred


# # Location Based Analysis

# In[52]:


features = ['lat','long','amt']


# In[53]:


X_train,X_test,y_train,y_test = train_test_split(train_data[features],train_data['is_fraud'],test_size=0.2,random_state=42)


# In[54]:


model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination based on your dataset
model.fit(X_train)


# In[55]:


predictions = model.predict(X_test)


# In[56]:


predictions[predictions == 1] = 0  # Normal transactions
predictions[predictions == -1] = 1  # Anomalies (potential fraud)


# In[57]:


print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))


# # Uniqueness in Fingerprint

# In[58]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[59]:


features = ['is_fraud', 'gender', 'lat', 'long', 'category']


# In[60]:


X = train_data[features]
y=train_data['is_fraud']


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[67]:


model.fit(X_train, y_train)


# In[68]:


y_pred = model.predict(X_test)


# In[33]:


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[ ]:




