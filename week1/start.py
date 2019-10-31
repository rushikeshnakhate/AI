import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from docutils.nodes import block_quote


def open_file():
    tr_path = r'D:\AI\week2\data\train.csv'
    return pd.read_csv(tr_path)


def plot(df):
    x = df['SalePrice']
    y = df['GrLivArea']
    plt.scatter(x, y, marker="+")
    plt.title("sample graph")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show(block=True)
    df.plot('YearBuilt', 'SalePrice', kind='scatter', marker="+")


# GRADED
# Build a function that takes as input a matrix
# return the inverse of that matrix
# assign function to "inverse_of_matrix"
# YOUR ANSWER BELOW

def inverse_of_matrix(mat):
    """Calculate and return the multiplicative inverse of a matrix.

    Positional argument:
        mat -- a square matrix to invert

    Example:
        sample_matrix = [[1, 2], [3, 4]]
        the_inverse = inverse_of_matrix(sample_matrix)

    Requirements:
        This function depends on the numpy function `numpy.linalg.inv`.
    """
    return np.linalg.inv(mat)


# GRADED
# Build a function  called "least_squares_weights"
# take as input two matricies corresponding to the X inputs and y target
# assume the matricies are of the correct dimensions

# Step 1: ensure that the number of rows of each matrix is greater than or equal to the number
# of columns.
# # If not, transpose the matricies.
# In particular, the y input should end up as a n-by-1 matrix, and the x input as a n-by-p matrix

# Step 2: *prepend* an n-by-1 column of ones to the input_x matrix

# Step 3: Use the above equation to calculate the least squares weights.

# NB: `.shape`, `np.matmul`, `np.linalg.inv`, `np.ones` and `np.transpose` will be valuable.
# If those above functions are used, the weights should be accessable as below:  
# weights = least_squares_weights(train_x, train_y)
# weight1 = weights[0][0]; weight2 = weights[1][0];... weight<n+1> = weights[n][0]

# YOUR ANSWER BELOW


def least_squares_weights(input_x, target_y):
    """Calculate linear regression least squares weights.

    Positional arguments:
        input_x -- matrix of training input data
        target_y -- vector of training output values

        The dimensions of X and y will be either p-by-n and 1-by-n
        Or n-by-p and n-by-1

    Example:
        import numpy as np
        training_y = np.array([[208500, 181500, 223500, 
                                140000, 250000, 143000, 
                                307000, 200000, 129900, 
                                118000]])
        training_x = np.array([[1710, 1262, 1786, 
                                1717, 2198, 1362, 
                                1694, 2090, 1774, 
                                1077], 
                               [2003, 1976, 2001, 
                                1915, 2000, 1993, 
                                2004, 1973, 1931, 
                                1939]])
        weights = least_squares_weights(training_x, training_y)

        print(weights)  #--> np.array([[-2.29223802e+06],
                           [ 5.92536529e+01],
                           [ 1.20780450e+03]])

        print(weights[1][0]) #--> 59.25365290008861

    Assumptions:
        -- target_y is a vector whose length is the same as the
        number of observations in training_x
    """
    x_rows = len(input_x[:, 0])
    x_col = len(input_x[0, :])
    if x_rows < x_col:
        input_x = np.transpose(input_x)
    x = np.insert(input_x, 0, 1, axis=1)
    xT = np.transpose(x)
    first = np.matmul(np.linalg.inv(np.matmul(xT, x)), xT)
    y_rows = len(target_y[:, 0])
    y_col = len(target_y[0, :])
    if y_rows < y_col:
        target_y = np.transpose(target_y)
    return np.matmul(first, target_y)


# GRADED
# Build a function called "column_cutoff"
# As inputs, accept a Pandas Dataframe and a list of tuples.
# Tuples in format (column_name, min_value, max_value)
# Return a DataFrame which excludes rows where the value in specified column exceeds "max_value"
# or is less than "min_value".
# # NB: DO NOT remove rows if the column value is equal to the min/max value
# YOUR ANSWER BELOW

def column_cutoff(data_frame, cutoffs):
    """Subset data frame by cutting off limits on column values.

    Positional arguments:
        data -- pandas DataFrame object
        cutoffs -- list of tuples in the format: 
        (column_name, min_value, max_value)

    Example:
        data_frame = read_into_data_frame('train.csv')
        # Remove data points with SalePrice < $50,000
        # Remove data points with GrLiveAre > 4,000 square feet
        cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
        selected_data = column_cutoff(data_frame, cutoffs)
    """
    data_subset = data_frame
    for col, min, max in cutoffs:
        data_subset = data_subset.loc[data_subset[col] >= min, :]
        data_subset = data_subset.loc[data_subset[col] <= max, :]
    return data_subset


def plot_real_data(df):
    df_sub = df[['SalePrice', 'GrLivArea', 'YearBuilt']]
    cutoff = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
    df_sub_cutoff = column_cutoff(df_sub, cutoff)
    X = df_sub_cutoff['GrLivArea'].values
    Y = df_sub_cutoff['SalePrice'].values
    training_x = np.array([X])
    training_y = np.array([Y])
    weights = least_squares_weights(training_x, training_y)
    max_X = np.max(X) + 500
    min_X = np.max(X) - 500
    reg_x = np.linspace(min_X, max_X, 1000)
    reg_y = weights[0][0] + weights[1][0] * reg_x
    plt.plot(reg_x, reg_y, color='#58b970', label='regression line')
    plt.scatter(X, Y, c='k', label='DATA')
    plt.xlabel('GrLivArea')
    plt.ylabel('SalePrice')
    plt.legend()
    plt.show()


def cal_RMSE(df):
    ###copied these lines form above function as
    df_sub = df[['SalePrice', 'GrLivArea', 'YearBuilt']]
    cutoff = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
    df_sub_cutoff = column_cutoff(df_sub, cutoff)
    X = df_sub_cutoff['GrLivArea'].values
    Y = df_sub_cutoff['SalePrice'].values
    training_x = np.array([X])
    training_y = np.array([Y])
    weights = least_squares_weights(training_x, training_y)
    ##### to get the weight
    b_0 = weights[0][0]
    b_1 = weights[1][0]

    rmse =0
    for i in range(len(Y)):
        y_pread = b_0 + b_1 * X[i]
        rmse += ((Y[i] - y_pread) ** 2)

    rmse = np.sqrt(rmse / len(Y))
    print(rmse)


if __name__ == '__main__':
    data = open_file()
    plot(data)
    plot_real_data(data)
    cal_RMSE(data)
