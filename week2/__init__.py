import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
tr_path = r'D:\AI\week2\data\train.csv'


def read_data():
    return pd.read_csv(tr_path)


def plot(data):
    data.plot('GrLivArea', 'SalePrice', kind='scatter', marker='x')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('testing from function ')
    plt.show(block=True)


def get_subset(data):
    return data.ix[:, 'Street']
    print(subset)


df = read_data()
plot(df)
get_subset(df)
