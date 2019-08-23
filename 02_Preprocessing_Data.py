# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


import seaborn as sns

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd
pd.set_option('display.max_rows', 30)

df = pd.read_csv("Toddler_Autism_dataset_July_2018.csv")


# -

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


missing_values_table(df)


# # Great, no nan values! - now let's remove the possible outliers
#
# # First let's see what the box plots look like first

# +
def simple_box_plot(df, x_attrib, y_attrib):
    f, axes = plt.subplots(ncols=1, figsize=(7,7))

    sns.boxplot(x=y_attrib, y=x_attrib, data=df)
    axes.set_title(x_attrib)

    
    for i in df[y_attrib].unique():
        print("Median for '{}': {}".format(i, df[x_attrib][df[y_attrib] == i].median()))

plt.show()
    
# -

def remove_outliers(df, x_attrib, y_attrib):

    for i in df[y_attrib].unique():
        
        m, n = df.shape
        print('Number of rows: {}'.format(m))
        
        remove_list = df[x_attrib].loc[df[y_attrib] == i].values
        q25, q75 = np.percentile(remove_list, 25), np.percentile(remove_list, 75)
        print('Lower Quartile: {} | Upper Quartile: {}'.format(q25, q75))
        iqr = q75 - q25
        print('iqr: {}'.format(iqr))

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        print('Cut Off: {}'.format(cut_off))
        print('Lower Extreme: {}'.format(lower))
        print('Upper Extreme: {}'.format(upper))

        outliers = [x for x in remove_list if x < lower or x > upper]
        print('Number of Outliers for {} Cases: {}'.format(i, len(outliers)))
        print('outliers:{}'.format(outliers))

        for d in outliers:
            #delete_row = new_df[new_df[y_attrib]==i].index
            #new_df = new_df.drop(delete_row)
            df = df[df[x_attrib] != d]
        
        m, n = df.shape
        print('Number of rows for new dataframe: {}\n'.format(m))
    
    new_df = df
    
    print('----' * 27)
    return new_df


df.head()

simple_box_plot(df, 'Age_Mons', 'Class/ASD Traits ')

df.columns

simple_box_plot(df, 'Qchat-10-Score', 'Class/ASD Traits ')

# # These look okay to me

# +
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, train_size= 0.7, random_state = 42)

y_train = train_df['Class/ASD Traits ']
X_train = train_df.drop(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
                         'Class/ASD Traits ', 'Case_No', 'Who completed the test'], axis=1)

y_test = test_df['Class/ASD Traits ']
X_test = test_df.drop(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
                       'Class/ASD Traits ', 'Case_No', 'Who completed the test'], axis=1)


# -

X_train.to_csv(r'X_train.csv')
X_test.to_csv(r'X_test.csv')
y_train.to_csv(r'y_train.csv')
y_test.to_csv(r'y_test.csv')

# # Now let's use pipelines to scale the numerical data and one hot encode the categorical data

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class selector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
      
num_attributes = ['Age_Mons',
       'Qchat-10-Score']

num_pipeline = Pipeline([
            ('selector', selector(num_attributes)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
                    ])

cat_attributes = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']

cat_pipeline = Pipeline([
                ('selector', selector(cat_attributes)),
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('cat_encoder', OneHotEncoder(sparse=False)),
])

# +
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# +
X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)


# note that we are using transform() on X_pretest 
#we standardize our training dataset, we need to keep the parameters (mean and standard deviation for each feature). 
#Then, we use these parameters to transform our test data and any future data later on
#fit() just calculates the parameters (e.g. 𝜇 and 𝜎 in case of StandardScaler) 
#and saves them as an internal objects state. 
#Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
X_test_processed = full_pipeline.transform(X_test)
X_test_processed = pd.DataFrame(X_test_processed)

# -

X_train_processed.to_csv(r'X_train_processed.csv')
X_test_processed.to_csv(r'X_test_processed.csv')


