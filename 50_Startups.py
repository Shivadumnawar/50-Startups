# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:47:05 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('50_Startups.csv')

df.info()

df.describe()

df.isnull().sum() 
# no null values

# check outliers
df.plot(kind='box')

X= df.iloc[:, :-1]
y= df.iloc[:, -1].values.reshape(-1,1)

# one hot encoding
X= pd.get_dummies(X, columns=['State'], drop_first=True)

# correlation
c= df.corr()
plt.figure(figsize=(12,9))
sns.heatmap(c, cmap='coolwarm', annot=True)
plt.yticks(rotation=0)
plt.tight_layout()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=75)

# standardization
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

X_train= ss.fit_transform(X_train)
X_test= ss.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train, y_train)

pred= reg.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test, pred)

mean_squared_error(y_test, pred)










