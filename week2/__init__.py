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


def standardize(num_list):
    """
    Standardize the given list of numbers

    Positional arguments:
        num_list -- a list of numbers

    Example:
        num_list = [1,2,3,3,4,4,5,5,5,5,5]
        nl_std = standardize(num_list)
        print(np.round(nl_std,2))
        #--> np.array([-2.11, -1.36, -0.61, -0.61,
                0.14,  0.14,  0.88,  0.88,  0.88,
                0.88,  0.88])

    NOTE: the sample standard deviation should be calculated with 0 "Delta Degrees of Freedom"
    """
    mean = np.mean(num_list)
    std_dev = np.std(num_list)
    subtract_mean = num_list - mean
    return subtract_mean / std_dev


def preprocess_for_regularization(data, y_column_name, x_column_names):
    """
    Perform mean subtraction and dimension standardization on data

    Positional argument:
        data -- a pandas dataframe of the data to pre-process
        y_column_name -- the name (string) of the column that contains
            the target of the training data.
        x_column_names -- a *list* of the names of columns that contain the
            observations to be standardized

    Returns:
        Return a DataFrame consisting only of the columns included
        in `y_column_name` and `x_column_names`.
        Where the y_column has been mean-centered, and the
        x_columns have been mean-centered/standardized.


    Example:
        data = pd.read_csv(tr_path).head()
        prepro_data = preprocess_for_regularization(data,'SalePrice', ['GrLivArea','YearBuilt'])

        print(prepro_data) #-->
                   GrLivArea  YearBuilt  SalePrice
                0  -0.082772   0.716753     7800.0
                1  -1.590161  -0.089594   -19200.0
                2   0.172946   0.657024    22800.0
                3  -0.059219  -1.911342   -60700.0
                4   1.559205   0.627159    49300.0

    NOTE: The sample standard deviation should be calculated with 0 "Delta Degrees of Freedom"

    If your answer does not match the example answer,
    check the default degrees of freedom in your standard deviation function.
    """
    y_data = data[[y_column_name]]
    y_mean = np.mean(y_data)
    y_centered = y_data - y_mean;

    x_data = data.loc[:, x_column_names]
    x_mean = np.mean(x_data)
    x_std_dev = np.std(x_data)
    x_standardized = (x_data - x_mean) / x_std_dev
    return pd.concat([x_standardized, y_centered.reindex(x_standardized.index)], axis=1)


if __name__ == '__main__':
    data = read_data()
    num_list = [1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5]
    nl_std = standardize(num_list)
    print(np.round(nl_std, 2))
    prepro_data = preprocess_for_regularization(data.head(), 'SalePrice', ['GrLivArea', 'YearBuilt'])
