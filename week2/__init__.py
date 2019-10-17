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


### GRADED
### Code a function called "ridge_regression_weights"
### ACCEPT three inputs:
### Two matricies corresponding to the x inputs and y target
### and a number (int or float) for the lambda parameter

### RETURN a numpy array of regression weights

### The following must be accomplished:

### Ensure the number of rows of each the X matrix is greater than the number of columns.
### ### If not, transpose the matrix.
### Ultimately, the y input will have length n.
### Thus the x input should be in the shape n-by-p

### *Prepend* an n-by-1 column of ones to the input_x matrix

### Use the above equation to calculate the least squares weights.
### This will involve creating the lambda matrix---
### ### a p+1-by-p+1 matrix with the "lambda_param" on the diagonal
### ### p+1-by-p+1 because of the prepended "ones".

### NB: Pay close attention to the expected format of the returned
### weights. It is different / simplified from Assignment 1.

### YOUR ANSWER BELOW

def ridge_regression_weights(input_x, output_y, lambda_param):
    """Calculate ridge regression least squares weights.

    Positional arguments:
        input_x -- 2-d matrix of input data
        output_y -- 1-d numpy array of target values
        lambda_param -- lambda parameter that controls how heavily
            to penalize large weight values

    Example:
        training_y = np.array([208500, 181500, 223500,
                                140000, 250000, 143000,
                                307000, 200000, 129900,
                                118000])

        training_x = np.array([[1710, 1262, 1786,
                                1717, 2198, 1362,
                                1694, 2090, 1774,
                                1077],
                               [2003, 1976, 2001,
                                1915, 2000, 1993,
                                2004, 1973, 1931,
                                1939]])
        lambda_param = 10

        rrw = ridge_regression_weights(training_x, training_y, lambda_param)

        print(rrw) #--> np.array([-576.67947107,   77.45913349,   31.50189177])
        print(rrw[2]) #--> 31.50189177

    Assumptions:
        -- output_y is a vector whose length is the same as the
        number of observations in input_x
        -- lambda_param has a value greater than 0
    """

    x_rows = len(input_x[:, 0])
    x_col = len(input_x[0, :])
    if (x_rows < x_col):
        input_x = np.transpose(input_x)

    ones_col = np.ones(len(input_x[:, 0]))
    input_x = np.insert(input_x, 0, ones_col, axis=1)

    xT = np.transpose(input_x)

    lambda_plus_x = lambda_param + np.matmul(xT, input_x)
    lambda_plus_x_inv = np.linalg.inv(lambda_plus_x)
    return np.matmul(np.matmul(lambda_plus_x_inv, xT), output_y)


def hidden(hp):
    if (hp <= 0) or (hp >= 50):
        print("input provided is out of bound")
    nums = np.logspace(0, 5, num=1000)
    vals = nums ** 43.123985172351235134687934
    user_vals = nums ** hp
    return (vals - user_vals)


### GRADED
### Code a function called "minimize"
### ACCEPT one input: a function.

### That function will be similar to `hidden` created above and available for your exploration.
### Like 'hidden', the passed function will take a single argument, a number between 0 and 50 exclusive
### and then, the function will return a numpy array of 1000 numbers.

### RETURN the value that makes the mean of the array returned by 'passed_func' as close to 0 as possible

### Note, you will almost certainly NOT be able to find the number that makes the mean exactly 0
### YOUR ANSWER BELOW

def minimize(passed_func):
    """
    Find the numeric value that makes the mean of the
    output array returned from 'passed_func' as close to 0 as possible.

    Positional Argument:
        passed_func -- a function that takes a single number (between 0 and 50 exclusive)
            as input, and returns a list of 1000 floats.

    Example:
        passed_func = hidden
        min_hidden = minimize(passed_func)
        print(round(min_hidden,4))
        #--> 43.1204 (answers will vary slightly, must be close to 43.123985172351)

    """
    # Create values to test
    test_vals = passed_func(43.1204)

    # Find mean of returned array from function
    ret_vals = np.mean(test_vals)

    # Find smallest mean
    min_mean = ret_vals.min()
    return min_mean

    # Return the test value that creates the smallest mean
    # return ...


def lambda_search_func(lambda_param):
    # Define X and y
    # with preprocessing
    df = preprocess_for_regularization(data.head(50), 'SalePrice', ['GrLivArea', 'YearBuilt'])

    y_true = df['SalePrice'].values
    X = df[['GrLivArea', 'YearBuilt']].values

    # Calculate Weights then use for predictions
    weights = ridge_regression_weights(X, y_true, lambda_param)
    y_pred = weights[0] + np.matmul(X, weights[1:])

    # Calculate Residuals
    resid = y_true - y_pred

    # take absolute value to tune on mean-absolute-deviation
    # Alternatively, could use:
    # return resid **2-S
    # for tuning on mean-squared-error

    return abs(resid)


def sklearn():
    """
     Ridge Regression in `sklearn`
    Below gives the syntax for implementing ridge regression in sklearn.

    from sklearn.linear_model import Ridge, LinearRegression

    ### Note, the "alpha" parameter defines regularization strength.
    ### Lambda is a reserved word in `Python` -- Thus "alpha" instead

    ### An alpha of 0 is equivalent to least-squares regression
    """

    lr = LinearRegression()
    reg = Ridge(alpha=100000)
    reg0 = Ridge(alpha=0)

    # Notice how the consistent sklearn syntax may be used to easily fit many kinds of models
    for m, name in zip([lr, reg, reg0], ["LeastSquares", "Ridge alpha = 100000", "Ridge, alpha = 0"]):
        m.fit(data[['GrLivArea', 'YearBuilt']], data['SalePrice'])
        print(name, "Intercept:", m.intercept_, "Coefs:", m.coef_, "\n")


if __name__ == '__main__':
    data = read_data()
    num_list = [1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5]
    nl_std = standardize(num_list)
    prepro_data = preprocess_for_regularization(data.head(), 'SalePrice', ['GrLivArea', 'YearBuilt'])
    training_y = np.array([208500, 181500, 223500,
                           140000, 250000, 143000,
                           307000, 200000, 129900,
                           118000])

    training_x = np.array([[1710, 1262, 1786,
                            1717, 2198, 1362,
                            1694, 2090, 1774,
                            1077],
                           [2003, 1976, 2001,
                            1915, 2000, 1993,
                            2004, 1973, 1931,
                            1939]])
    lambda_param = 10
    rrw = ridge_regression_weights(training_x, training_y, lambda_param)
    ret = hidden(10)
    passed_func = hidden
    min_hidden = minimize(passed_func)
    sklearn()
