# Data Preprocessing Template

## Part 1: Data Preprocessing
### Get the Dataset
### Importing the Libraries
### Importing the Dataset
### Missing Data
### Categorical Data
### Splitting the Dataset into the Training Set and Test Set
### Feature Scaling
### Data Preprocessing Template


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Importing the dataset
from os import chdir, getcwd
chdir('\\\\wil-entsasprd06\DigitalAnalytics\D17911_Lu\Machine Learning\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------')
getcwd()
dataset = pd.read_csv('Data.csv')
dataset.head(5)
X = dataset.iloc[:, :-1].values   ## array
X
## Compare with dataset.iloc[:, :-1]  --  dataframe
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set, sklearn.cross_validation, sklearn.model_selection

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
## random_state = 42

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


## Why scaling is important: Machine learning models are based on Euclidean distance, 
## Euclidean distance will be dominated by the large scale variable

## Methods of Feature Scaling
## Standardization: X = (x-mean)/standard deviation
## Normalization (MinMaxScale) x = (x-min)/(max-min)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Question: Do we need to fit and transform dummy variables (y)?
## It depends on your context, it depends on how much you want to keep interpretation in your models

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


## Though some machine learning models are not depend on Euclidean Distance, we still need to feature scaling
## Because algorithms will converge much faster, for example, Decision Trees

## In classification problem, we do not need to scale dependent variable; In regression problem, we can do that.

















