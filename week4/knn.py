import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_feature_names():
    features_names = pd.read_table(r'D:\AI\week4\data\HAR_Dataset\features.txt', sep='\n', header=None)
    return features_names


def read_train_data():
    har_train_data = pd.read_table(r'D:\AI\week4\data\HAR_Dataset\train\X_train.txt', sep='\s+', header=None)
    row = len(har_train_data.index)
    col = len(har_train_data.columns)
    return har_train_data


def read_train_labels():
    return pd.read_table(r'D:\AI\week4\data\HAR_Dataset\train\y_train.txt', sep='\n', header=None, names=["label"],
                         squeeze=True)


def plot_correlation_first_twenty_feature(har_train):
    first_twenty = har_train.iloc[:, :20]
    correlation_matrix = first_twenty.corr()
    mask = np.zeros_like(correlation_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0, linewidths=.5, cbar_kws={'shrink': 0.5})
    plt.show(block=True)


def find_euclidean_distance(p1, p2):
    a1 = np.array(p1)
    a2 = np.array(p2)
    return np.sqrt(np.sum((a1 - a2) ** 2))


# GRADED
# Code a function called "all_distances"
# ACCEPT two inputs:
# An observation from a data set.  e.g: har_train.iloc[50,:]
# The full data set. e.g. har_train.

# Create a <list> or numpy array of distances between:
# # that single point, and all points in the full dataset

# RETURN the list of distances SORTED from smallest to largest.

# Notes:
# Use `np.linalg.norm()`, as described in above cell.
# The smallest distance should be 0.

# YOUR ANSWER BELOW


def all_distances(test_point, data_set):
    """
    Find and return a list of distances between the "test_point"
    and all the points in "data_set", sorted from smallest to largest.

    Positional Arguments:
        test_point -- a Pandas Series corresponding to a row in "data_set"
        data_set -- a Pandas DataFrame

    Example:
        test_point = har_train.iloc[50,:]
        data_set = har_train

        print(all_distances(test_point, data_set)[:5])
        #--> [0.0, 2.7970187358249854, 2.922792670143521, 2.966555149052483, 3.033982453218797]

    """
    return np.sort(np.apply_along_axis(np.linalg.norm, 1, (test_point - data_set)))


# GRADED
# Code a function called "labels_of_smallest"
# ACCEPT three inputs:
# 1&2: numpy arrays, corresponding to 1: a numeric column and 2: a label column.
# # The i-th member of the numeric column corresponds to the i-th member of the label column
# 3: an integer (>0); n.

# RETURN a list (or numpy array) of the n labels corresponding to 
# # the n smallest values in the numeric column.
# NOTE: Make sure the order of labels corresponds to the order of values.

# Hint: The labels are found in har_train_labels or y
# Hint: `pd.concat()` might be useful for this or subsequent exercisces  
# YOUR ANSWER BELOW

def labels_of_smallest(numeric, labels, n):
    """
    Return the n labels corresponding to the n smallest values in the "numeric"
    numpy array.

    Positional Arguments:
        numeric -- a numpy array of numbers
        labels -- a numpy array of labels (string or numeric)
            corresponding to the values in "numeric"
        n -- a positive integer

    Example:
        numeric = np.array([7,6,5,4,3,2,1])
        labels = np.array(["a","a","b","b","b","a","a"])
        n = 6

        print(labels_of_smallest(numeric, labels, n))
        #--> np.array(['a', 'a', 'b', 'b', 'b', 'a'])
    """
    df = pd.DataFrame({'numeric': numeric, 'labels': labels})
    labels_df = df.sort_values(by=['numeric'], ascending=True)['labels'].head(n)
    return labels_df.values


# GRADED
from collections import Counter


# Build a function called "label_voting"
# ACCEPT a non-empty numpy array of labels as input
# RETURN the value that appears most frequently in that array
# In the case of of a tie, RETURN the value in the tie that appears first in the array


# YOUR ANSWER BELOW

def label_voting(labels):
    """
    Given a numpy array of labels. Return the label that appears most frequently
    If there is a tie for most frequent, return the label that appears first.

    Positional Argument:
        labels -- a numpy array of labels

    Example:
        lab1 = np.array([1,2,2,3,3])
        lab2 = np.array(["a","a","b","b","b"])

        print(label_voting(lab1)) #--> 2
        print(label_voting(lab2)) #--> "b"

    """


if __name__ == '__main__':
    feats = read_feature_names()
    har_train = read_train_data()
    har_train_labels = read_train_labels()
    har_train.columns = feats.iloc[:, 0]
    plot_correlation_first_twenty_feature(har_train)
    read_train_data()
