# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:34:11 2024

@author: arrja
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearnex import patch_sklearn
import graphviz



df = pd.read_csv('./hotel_bookings.csv')
pd.set_option('display.max_columns', None)

## Drop reservation_status_date and country
df = df.drop('reservation_status_date', axis=1)
df = df.drop('reservation_status', axis=1)
df = df.drop('country', axis=1)
df = df.drop('assigned_room_type', axis=1)

## Convert binary columns from int to binary
df['is_canceled'] = df.is_canceled.apply(lambda x: format(int(x), 'b'))
df['is_repeated_guest'] = df.is_canceled.apply(lambda x: format(int(x), 'b'))

## Convert 'NA' values to 0
## Convert agent column into binary
df['agent'].fillna(value=0, inplace=True)
df['agent'] = df['agent'].astype(np.int64)
df.loc[df['agent'] > 0, 'agent'] = 1
df['agent'] = df.agent.apply(lambda x: format(int(x), 'b'))

## Convert 'NA' values to 0
## Convert company column into binary
df['company'].fillna(value=0, inplace=True)
df['company'] = df['company'].astype(np.int64)
df.loc[df['company'] > 0, 'company'] = 1
df['company'] = df.company.apply(lambda x: format(int(x), 'b'))

## Create total_occupancy column
## Convert 'NA' values to 0
## Convert children column into int
df['children'].fillna(value=0, inplace=True)
df['children'] = df['children'].astype(np.int64)
df['total_occupancy'] = df['adults'] + df['children'] + df['babies']
df = df.drop('adults', axis=1)
df = df.drop('children', axis=1)
df = df.drop('babies', axis=1)

## Create length_of_stay column
df['length_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df = df.drop('stays_in_weekend_nights', axis=1)
df = df.drop('stays_in_week_nights', axis=1)

## Create reservation_date column
#print(df['arrival_date_day_of_month'].dtypes)
df['reservation_date'] = pd.to_datetime(df[['arrival_date_day_of_month','arrival_date_month','arrival_date_year']]
                   .astype(str).apply(' '.join, 1), format='%d %B %Y')
df = df.drop('arrival_date_day_of_month', axis=1)
df = df.drop('arrival_date_month', axis=1)
df = df.drop('arrival_date_year', axis=1)
df['reservation_date']=df['reservation_date'].map(dt.datetime.toordinal)

## Remove adr outliers
df = df[df['adr'] >= 0]
df = df[df['adr'] <= 1000]

## Remove total_occupancy outliers
df = df[df['total_occupancy'] <= 8]
df = df[df['total_occupancy'] > 0]

sns.displot(df, x="deposit_type", col="is_canceled")

## Convert categorical columns to categories
df['hotel'] = df['hotel'].astype('category').cat.codes
df['meal'] = df['meal'].astype('category').cat.codes
df['market_segment'] = df['market_segment'].astype('category').cat.codes
df['distribution_channel'] = df['distribution_channel'].astype('category').cat.codes
df['reserved_room_type'] = df['reserved_room_type'].astype('category').cat.codes
df['deposit_type'] = df['deposit_type'].astype('category').cat.codes
df['customer_type'] = df['customer_type'].astype('category').cat.codes

## Partition Data
Y = df['is_canceled']
X = df.drop('is_canceled', axis=1)
X = X.drop('is_repeated_guest', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

feature_names = x_train.columns
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


## Perform logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

## Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

perm_importance = permutation_importance(lr, x_test, y_test)
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
lrp = plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])


## Decision tree classifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=2)
dt.fit(x_train, y_train)
predictions = dt.predict(x_test)

## Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

perm_importance = permutation_importance(dt, x_test, y_test)
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
dtp = plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])

class_names = ['not canceled', 'canceled']
tree.plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)


## Random forest classifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

## Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

perm_importance = permutation_importance(rf, x_test, y_test)
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
rfp = plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])


## KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

# Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

perm_importance = permutation_importance(knn, x_test, y_test)
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
knnp = plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])


## SVM
patch_sklearn()

svm = svm.SVC(kernel='rbf', C = 1.0)
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)

# Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

perm_importance = permutation_importance(svm, x_test, y_test)
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
svmp = plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])


## MLP
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,
                    activation = 'relu',solver='adam',random_state=1)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

# Get metrics
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, average="macro"))
print(metrics.recall_score(y_test, predictions, average="macro"))
print(metrics.confusion_matrix(y_test, predictions))

