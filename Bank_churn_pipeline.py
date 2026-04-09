#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as gb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[106]:


pd.read_csv("Customer-Churn-Records.csv")
data


# In[107]:


data.info()


# In[108]:


data.shape


# In[109]:


data.isnull().sum()


# In[110]:


data = data.drop(['RowNumber','CustomerId','Surname'],axis = 1)


# In[111]:


lst = data.columns
for feature in lst:
    value = data[feature].value_counts()
    print(value)
    print('    ')
    


# In[112]:


Categorical_features =['Geography','Gender','Card Type']
Numerical_features = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Complain','Satisfaction Score','Point Earned']


# In[113]:


X = data.drop(["Exited"],axis = 1)
Y = data["Exited"]


# In[114]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)


# In[115]:


Preprocessor = ColumnTransformer(
    transformers = [("num" , StandardScaler(),Numerical_features),
                    ("Cat" , OneHotEncoder(handle_unknown='ignore'),Categorical_features)
                   ])


# In[116]:


models = {"Logistic Regression": LogisticRegression(),
          "XGboost Classifier": gb.XGBClassifier(n_estimators = 200,learning_rate=0.01,max_depth = None),
          "KNN Classifier" : KNeighborsClassifier(metric = "euclidean",n_neighbors = 10),
          "RF Classifier" : RandomForestClassifier(n_estimators = 600,max_depth=None,min_samples_split=4,min_samples_leaf=2,max_features='sqrt',oob_score=True,random_state=42)
         }


# In[117]:


pipelines = {name: Pipeline([('preprocessing',Preprocessor),('model',models)]) for name, models in models.items()}


# In[118]:


scores ={}


# In[119]:


for name, pipe in pipelines.items():
    pipe.fit(X_train,Y_train)
    y_pred = pipe.predict(X_test)
    score = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    scores[name] = score
    print(f"{name}, accuracy_score : {score:.3f}")
    print(report)


# In[ ]:




