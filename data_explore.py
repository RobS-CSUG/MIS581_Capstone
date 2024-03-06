# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:16:34 2024

@author: arrja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./hotel_bookings.csv')
pd.set_option('display.max_columns', None)

df = df.drop('reservation_status_date', axis=1)

## Convert binary columns from int to binary
df['is_canceled'] = df.is_canceled.apply(lambda x: format(int(x), 'b'))
df['is_repeated_guest'] = df.is_canceled.apply(lambda x: format(int(x), 'b'))

## Convert 'NA' values to 0
## Convert agent column into string
df['agent'].fillna(value=0, inplace=True)
df['agent'] = df['agent'].astype(np.int64)
df['agent'] = df['agent'].astype(str)
print(df['agent'])

## Convert 'NA' values to 0
## Convert company column into string
df['company'].fillna(value=0, inplace=True)
df['company'] = df['company'].astype(np.int64)
df['company'] = df['company'].astype(str)
print(df['company'])

## Create total_occupancy column
## Convert 'NA' values to 0
## Convert children column into int
df['children'].fillna(value=0, inplace=True)
df['children'] = df['children'].astype(np.int64)
df['total_occupancy'] = df['adults'] + df['children'] + df['babies']
df = df.drop('adults', axis=1)
df = df.drop('children', axis=1)
df = df.drop('babies', axis=1)

print(df['total_occupancy'].describe())
sns.displot(df, x='total_occupancy')

## Create length_of_stay column
df['length_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df = df.drop('stays_in_weekend_nights', axis=1)
df = df.drop('stays_in_week_nights', axis=1)

print(df['length_of_stay'].describe())
sns.displot(df, x='length_of_stay')

## Create reservation_date column
#print(df['arrival_date_day_of_month'].dtypes)
df['reservation_date'] = pd.to_datetime(df[['arrival_date_day_of_month','arrival_date_month','arrival_date_year']]
                   .astype(str).apply(' '.join, 1), format='%d %B %Y')
df = df.drop('arrival_date_day_of_month', axis=1)
df = df.drop('arrival_date_month', axis=1)
df = df.drop('arrival_date_year', axis=1)

print(df['reservation_date'].describe())
sns.displot(df, x='reservation_date')

## Descriptive Statistics for all features
df.describe(include='all').to_csv("describe.csv")

corr = df.corr(method = 'pearson')
plt.figure(figsize=(10,8), dpi =500)
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show()

fig, ax =plt.subplots(1,2,figsize=(20,10))
a = sns.histplot(df["agent"], ax=ax[0])
b = sns.histplot(df["company"], ax=ax[1])
fig.show()



