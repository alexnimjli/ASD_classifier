# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


import numpy as np
import os

np.random.seed(42)

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd
pd.set_option('display.max_rows', 30)

df = pd.read_csv("Toddler_Autism_dataset_July_2018.csv")
# -

# Domain: Autistic Spectrum Disorder (ASD) is a neurodevelopmental condition associated with significant healthcare costs, and early diagnosis can significantly reduce these. Unfortunately, waiting times for an ASD diagnosis are lengthy and procedures are not cost effective. The economic impact of autism and the increase in the number of ASD cases across the world reveals an urgent need for the development of easily implemented and effective screening methods. Therefore, a time-efficient and accessible ASD screening is imminent to help health professionals and inform individuals whether they should pursue formal clinical diagnosis. The rapid growth in the number of ASD cases worldwide necessitates datasets related to behaviour traits. However, such datasets are rare making it difficult to perform thorough analyses to improve the efficiency, sensitivity, specificity and predictive accuracy of the ASD screening process. Presently, very limited autism datasets associated with clinical or screening are available and most of them are genetic in nature. Hence, we propose a new dataset related to autism screening of toddlers that contained influential features to be utilised for further analysis especially in determining autistic traits and improving the classification of ASD cases. In this dataset, we record ten behavioural features (Q-Chat-10) plus other individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.
#
# See the doc file description "Toddler data description" for the data variables
#
# Data Type: Predictive and Descriptive: Nominal / categorical, binary and continuous Task: Classification but can be used for clustering and association or feature assessment Attribute Type: Categorical, continuous and binary
# Area: Medical, health and social science Missing values? No Number of Instances (records in your data set): 1054 Number of Attributes (fields within each record): 18 including the class variable Attribute Information: For Further information about the attributes/feature see doc file attached
#
# Attributes: A1-A10: Items within Q-Chat-10 in which questions possible answers : “Always, Usually, Sometimes, Rarly & Never” items’ values are mapped to “1” or “0” in the dataset. For questions 1-9 (A1-A9) in Q-chat-10, if the respose was Sometimes / Rarly / Never “1” is assigned to the question (A1-A9). However, for question 10 (A10), if the respose was Always / Usually / Sometimes then “1” is assigned to that question. If the user obtained More than 3 Add points together for all ten questions. If your child scores more than 3 (Q-chat-10- score) then there is a potential ASD traits otherwise no ASD traits are observed. The remaining features in the datasets are collected from the “submit” screen in the ASDTests screening app. It should be noted that the class varaible was assigned automatically based on the score obtained by the user while undergoing the screening process using the ASDTests app.
#
# Cite the below relevant papers when using the data for analysis: 1) Tabtah, F. (2017). Autism Spectrum Disorder Screening: Machine Learning Adaptation and DSM-5 Fulfillment. Proceedings of the 1st International Conference on Medical and Health Informatics 2017, pp.1-6. Taichung City, Taiwan, ACM. 2) Thabtah, F. (2017). ASDTests. A mobile app for ASD screening. www.asdtests.com [accessed December 20th, 2017]. 3) Thabtah, F. (2017). Machine Learning in Autistic Spectrum Disorder Behavioural Research: A Review. Informatics for Health and Social Care Journal. 4) Thabtah F, Kamalov F., Rajab K (2018) A new computational intelligence approach to detect autistic features for autism screening. International Journal of Medical Infromatics, Volume 117, pp. 112-124.

df.head()

df.shape

df.isnull().any(axis=0)

df.isnull().sum()


# Let's analysis the data with a bar plot function first
#

def plot_bar_graphs(df, attribute, y):
    plt.figure(1)
    plt.subplot(131)
    df[attribute].value_counts(normalize=True).plot.bar(figsize=(22,4),title= attribute)

    crosstab = pd.crosstab(df[attribute], df[y])
    crosstab.div(crosstab.sum(1).astype(float), axis=0).plot.bar(stacked=True)
    crosstab.plot.bar(stacked=True)

    res = df.groupby([attribute, y]).size().unstack()
    tot_col = 0
    for i in range(len(df[y].unique())):
        tot_col = tot_col + res[res.columns[i]]

    for i in range(len(df[y].unique())):
        res[i] = (res[res.columns[i]]/tot_col)

    res = res.sort_values(by = [0], ascending = True)
    print(res)

    return


plot_bar_graphs(df, 'Jaundice', 'Class/ASD Traits ')

df.columns

plot_bar_graphs(df, 'Sex', 'Class/ASD Traits ')

plot_bar_graphs(df, 'Family_mem_with_ASD', 'Class/ASD Traits ')

plot_bar_graphs(df, 'Ethnicity', 'Class/ASD Traits ')

# +
dodger_blue = '#1E90FF'
crimson = '#DC143C'
lime_green = '#32CD32'
red_wine = '#722f37'
white_wine = '#dbdd46'

def plot_histograms(df, x_attribute, n_bins, x_max, y_attribute):

    #this removes the rows with nan values for this attribute
    df = df.dropna(subset=[x_attribute])

    print ("Mean: {:0.2f}".format(df[x_attribute].mean()))
    print ("Median: {:0.2f}".format(df[x_attribute].median()))

    df[x_attribute].hist(bins= n_bins, color= crimson)

    #this plots the mean and median
    plt.plot([df[x_attribute].mean(), df[x_attribute].mean()], [0, 100],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[x_attribute].median(), df[x_attribute].median()], [0, 100],
        color='black', linestyle='--', linewidth=2, label='median')

    plt.xlim(xmin=0, xmax = x_max)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.title(x_attribute)
    plt.legend(loc='best')
    plt.show()

    for i in ['Yes', 'No']:
        print ("{} Mean: {:0.2f}".format(i, df[df[y_attribute]==i][x_attribute].mean()))
        print ("{} Median: {:0.2f}".format(i, df[df[ y_attribute]==i][x_attribute].median()))

    for i, j in zip(['Yes', 'No'], ['r', 'b']):
        df[df[y_attribute]==i][x_attribute].hist(bins=n_bins, color =j, label=i)


    for i, j in zip(list(df[y_attribute].unique()), ['g', 'y']):
        plt.plot([df[df[y_attribute]==i][x_attribute].mean(), df[df[y_attribute]==i][x_attribute].mean()],
            [0, 50], color=j, linestyle='-', linewidth=2, label='mean')
        plt.plot([df[df[y_attribute]==i][x_attribute].median(), df[df[y_attribute]==i][x_attribute].median()],
            [0, 50], color=j, linestyle='--', linewidth=2, label='median')

    plt.xlim(xmin=0, xmax = x_max)

    plt.title(x_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.legend(loc='best')
    plt.show()
    return



# -

plot_histograms(df, 'Age_Mons', 10, 40, 'Class/ASD Traits ')



plot_histograms(df, 'Qchat-10-Score', 10, 10, 'Class/ASD Traits ')
