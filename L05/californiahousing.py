# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:23:23 2020

@author: MB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
%matplotlib inline

#read the data file
housing = pd.read_csv("housing.csv")

#Get in insight of Dataset
housing.head()
housing.info()

housing['ocean_proximity'].value_counts()

# Let's plot a histogram to get the feel of type of data we are dealing with
housing.hist(bins=30,figsize=(15,20))

#%%
#Find mean, median and divide

#statistics.median(housing['median_income'].value_counts())
#statistics.mean(housing['median_income'].value_counts())
median_income = housing['median_income'].value_counts()
median_income.mean()
median_income1 = housing['median_income']
median_income1.mean()
median_income1.median()
median_income1.std()

# Spørgsmål a og b L05:
#mean beregner gennemsnit og er derfor den bedste i dette tilfælde.

#%%
# Try to find columns having corelation
corr_mat = housing.corr()
corr_mat['median_house_value'].sort_values(ascending = False)

#Try to visualize the 'Median Income' Features.
housing['median_income'].hist(bins=30)

housing['median_income'].value_counts()

# Try to create some Strata from 'Median income'
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].value_counts()

# Stratified Sampling using Scikit-learn's StratifiedShuffleSplit Class
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]
##
# Let’s compare income category proportion in Stratified Sampling
def strata_compare(data):
    return data['income_cat'].value_counts()/len(data)

compare_props = pd.DataFrame({
    "Overall": strata_compare(housing),
    "Train Stratified": strata_compare(train_set),
    "Test Stratified": strata_compare(test_set),
   }).sort_index()
compare_props["Train Strat. %error"] = 100 * compare_props["Train Stratified"] / compare_props["Overall"] - 100
compare_props["Test Strat. %error"] = 100 * compare_props["Test Stratified"] / compare_props["Overall"] - 100
compare_props

#Dropping 'income_cat' columns
for set_ in (train_set,test_set):
    set_.drop('income_cat',axis=1,inplace=True)
    
    
#Housing values
#Try to visualize the 'housing value' Features.
housing['median_house_value'].hist(bins=10)    
    
#Visualize data
df1 = housing['median_income']
df2= housing['median_house_value']
result = pd.concat([df1, df2], axis=1).corr()
plt.matshow(result)
plt.colorbar()
plt.show()

dfhouse = housing[['median_income'], housing['median_house_value']]
df = train_set.copy()

plt.matshow(dfhouse.corr())
plt.colorbar()
plt.show()


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#%%
#Scatter plot
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=df["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


#Normaldistribution
sigma = housing['median_income'].std()

mu = sigma = housing['median_income'].mean()
s = np.random.normal(mu, sigma, 1000)

abs(mu - np.mean(s))

abs(sigma - np.std(s, ddof=1))

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()


