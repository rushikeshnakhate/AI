import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### GRADED
### Code a function called `calc_posterior`

### ACCEPT three inputs
### Two floats: the likelihood and the prior
### One list of tuples, where each tuple has two values corresponding to:
### ### ( P(Bn) , P(A|Bn) )
### ### ### Assume the list of tuples accounts for all potential values of B
### ### ### And that those values of B are all mutually exclusive.
### The list of tuples allows for the calculation of normalization constant.

### RETURN a float corresponding to the posterior probability

### YOUR ANSWER BELOW
def calc_posterior(likelihood, prior, norm_list̥):
    """
       Calculate the posterior probability given likelihood,
       prior, and normalization

       Positional Arguments:
           likelihood -- float, between 0 and 1
           prior -- float, between 0 and 1
           norm_list -- list of tuples, each tuple has two values
               the first value corresponding to the probability of a value of "b"
               the second value corresponding to the probability of
                   a value of "a" given that value of "b"
       Example:
           likelihood = .8
           prior = .3
           norm_list = [(.25 , .9), (.5, .5), (.25,.2)]
           print(calc_posterior(likelihood, prior, norm_list))
          ̥ # --> 0.45714285714285713
       """


### GRADED
### Build a function called "x_preprocess"
### ACCEPT one input, a numpy array
### ### Array may be one or two dimensions

### If input is two dimensional, make sure there are more rows than columns
### ### Then prepend a column of ones for intercept term
### If input is one-dimensional, prepend a one

### RETURN a numpy array, prepared as described above,
### which is now ready for matrix multiplication with regression weights

def x_preprocess(input_x):
    """
    Reshape the input (if needed), and prepend a "1" to every observation

    Positional Argument:
        input_x -- a numpy array, one- or two-dimensional

    Example:
        input1 = np.array([[2,3,6,9],[4,5,7,10]])
        input2 = np.array([2,3,6])
        input3 = np.array([[2,4],[3,5],[6,7],[9,10]])

        for i in [input1, input2, input3]:
            print(x_preprocess(i), "\n")

        # -->[[ 1.  2.  4.]
              [ 1.  3.  5.]
              [ 1.  6.  7.]
              [ 1.  9. 10.]]

            [1 2 3 6]

            [[ 1.  2.  4.]
             [ 1.  3.  5.]
             [ 1.  6.  7.]
             [ 1.  9. 10.]]

    Assumptions:
        Assume that if the input is two dimensional, that the observations are more numerous
            than the features, and thus, the observations should be the rows, and features the columns
    """
    ndim = input_x.ndim
    if ndim == 2:
        rows = len(input_x[:, 0])
        col = len(input_x[0, :])
        if rows < col:
            input_x = np.transpose(input_x)
        ones_col = np.ones(len(input_x[:, 0]))
        input_x = np.insert(input_x, 0, ones_col, axis=1)
    if ndim == 1:
        input_x = np.insert(input_x, 0, 1, axis=0)
    return input_x


### GRADED
### Build a function called `calculate_map_coefficients`

### ACCEPT four inputs:
### Two numpy arrays; an X-matrix and y-vector
### Two positive numbers, a lambda parameter, and value for sigma^2

### RETURN a 1-d numpy vector of weights.

### ASSUME your x-matrix has been preprocessed:
### observations are in rows, features in columns, and a column of 1's prepended.

### Use the above equation to calculate the MAP weights.
### ### This will involve creating the lambda matrix.
### ### The MAP weights are equal to the Ridge Regression weights

### NB: `.shape`, `np.matmul`, `np.linalg.inv`,
### `np.ones`, `np.identity` and `np.transpose` will be valuable.

### If either the "sigma_squared" or "lambda_param" are equal to 0, the return will be
### equivalent to ordinary least squares.

### YOUR ANSWER BELOW

def calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared):
    """
    Calculate the maximum a posteriori LR parameters

     Positional arguments:
        aug_x -- x-matrix of training input data, augmented with column of 1's
        output_y -- vector of training output values
        lambda_param -- positive number; lambda parameter that
            controls how heavily to penalize large coefficient values
        sigma_squared -- data noise estimate

    Example:
        output_y = np.array([208500, 181500, 223500,
                             140000, 250000, 143000,
                             307000, 200000, 129900,
                             118000])

        aug_x = np. array([[   1., 1710., 2003.],
                           [   1., 1262., 1976.],
                           [   1., 1786., 2001.],
                           [   1., 1717., 1915.],
                           [   1., 2198., 2000.],
                           [   1., 1362., 1993.],
                           [   1., 1694., 2004.],
                           [   1., 2090., 1973.],
                           [   1., 1774., 1931.],
                           [   1., 1077., 1939.]])

        lambda_param = 0.01

        sigma_squared = 1000

        map_coef = calculate_map_coefficients(aug_x, output_y,
                                             lambda_param, sigma_squared)

        ml_coef = calculate_map_coefficients(aug_x, output_y, 0,0)

        print(map_coef)
        # --> np.array([-576.67947107   77.45913349   31.50189177])

        print(ml_coef)
        #--> np.array([-2.29223802e+06  5.92536529e+01  1.20780450e+03])

    Assumptions:
        -- output_y is a vector whose length is the same as the
        number of rows in input_x
        -- input_x has more observations than it does features.
        -- lambda_param has a value greater than 0
    """
    coefs = np.array([])
    return coefs


### GRADED
### Build a function called `calculate_map_coefficients`

### ACCEPT four inputs:
### Two numpy arrays; an X-matrix and y-vector
### Two positive numbers, a lambda parameter, and value for sigma^2

### RETURN a 1-d numpy vector of weights.

### ASSUME your x-matrix has been preprocessed:
### observations are in rows, features in columns, and a column of 1's prepended.

### Use the above equation to calculate the MAP weights.
### ### This will involve creating the lambda matrix.
### ### The MAP weights are equal to the Ridge Regression weights

### NB: `.shape`, `np.matmul`, `np.linalg.inv`,
### `np.ones`, `np.identity` and `np.transpose` will be valuable.

### If either the "sigma_squared" or "lambda_param" are equal to 0, the return will be
### equivalent to ordinary least squares.

### YOUR ANSWER BELOW

def calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared):
    """
    Calculate the maximum a posteriori LR parameters

     Positional arguments:
        aug_x -- x-matrix of training input data, augmented with column of 1's
        output_y -- vector of training output values
        lambda_param -- positive number; lambda parameter that
            controls how heavily to penalize large coefficient values
        sigma_squared -- data noise estimate

    Example:
        output_y = np.array([208500, 181500, 223500,
                             140000, 250000, 143000,
                             307000, 200000, 129900,
                             118000])

        aug_x = np. array([[   1., 1710., 2003.],
                           [   1., 1262., 1976.],
                           [   1., 1786., 2001.],
                           [   1., 1717., 1915.],
                           [   1., 2198., 2000.],
                           [   1., 1362., 1993.],
                           [   1., 1694., 2004.],
                           [   1., 2090., 1973.],
                           [   1., 1774., 1931.],
                           [   1., 1077., 1939.]])

        lambda_param = 0.01

        sigma_squared = 1000

        map_coef = calculate_map_coefficients(aug_x, output_y,
                                             lambda_param, sigma_squared)

        ml_coef = calculate_map_coefficients(aug_x, output_y, 0,0)

        print(map_coef)
        # --> np.array([-576.67947107   77.45913349   31.50189177])

        print(ml_coef)
        #--> np.array([-2.29223802e+06  5.92536529e+01  1.20780450e+03])

    Assumptions:
        -- output_y is a vector whose length is the same as the
        number of rows in input_x
        -- input_x has more observations than it does features.
        -- lambda_param has a value greater than 0
    """
    xT = np.transpose(aug_x)
    lambda_mul_singma = lambda_param * sigma_squared;
    fist_inv = np.linalg.inv(lambda_mul_singma + np.matmul(xT, aug_x))
    coefs = np.matmul(np.matmul(fist_inv, xT), output_y)
    return coefs


### GRADED
### Code a function called `esimate_data_noise`

### ACCEPT three inputs, all numpy arrays
### One matrix coresponding to the augmented x-matrix
### Two vectors, one of the y-target, and one of ML weights.

### RETURN the empirical data noise estimate: sigma^2. Calculated with equation given above.

### NB: "n" is the number of observations in X (rows)
### "d" is the number of features in aug_x (columns)

### YOUR ANSWER BELOW

def estimate_data_noise(aug_x, output_y, weights):
    """Return empirical data noise estimate \sigma^2
    Use the LR weights in the equation supplied above

    Positional arguments:
        aug_x -- matrix of training input data
        output_y -- vector of training output values
        weights -- vector of LR weights calculated from output_y and aug_x


    Example:
        output_y = np.array([208500, 181500, 223500,
                                140000, 250000, 143000,
                                307000, 200000, 129900,
                                118000])
        aug_x = np. array([[   1., 1710., 2003.],
                           [   1., 1262., 1976.],
                           [   1., 1786., 2001.],
                           [   1., 1717., 1915.],
                           [   1., 2198., 2000.],
                           [   1., 1362., 1993.],
                           [   1., 1694., 2004.],
                           [   1., 2090., 1973.],
                           [   1., 1774., 1931.],
                           [   1., 1077., 1939.]])

        ml_weights = calculate_map_coefficients(aug_x, output_y, 0, 0)

        print(ml_weights)
        # --> [-2.29223802e+06  5.92536529e+01  1.20780450e+03]

        sig2 = estimate_data_noise(aug_x, output_y, ml_weights)
        print(sig2)
        #--> 1471223687.1593

    Assumptions:
        -- training_input_y is a vector whose length is the same as the
        number of rows in training_x
        -- input x has more observations than it does features.
        -- lambda_param has a value greater than 0
    """
    n = len(aug_x[:, 0])
    d = len(aug_x[0, :])
    Xw = aug_x * weights
    data_noise = sum(range(np.square(output_y[n] - Xw)) ** 2) / (n - d)
    print(data_noise)


if __name__ == '__main__':
    likelihood = .8
    prior = .3
    norm_list = [(.25, .9), (.5, .5), (.25, .2)]
    calc_posterior(likelihood, prior, norm_list)
    input1 = np.array([[2, 3, 6, 9], [4, 5, 7, 10]])
    input2 = np.array([2, 3, 6])
    input3 = np.array([[2, 4], [3, 5], [6, 7], [9, 10]])
    x_preprocess(input3)
    output_y = np.array([208500, 181500, 223500,
                         140000, 250000, 143000,
                         307000, 200000, 129900,
                         118000])

    aug_x = np.array([[1., 1710., 2003.],
                      [1., 1262., 1976.],
                      [1., 1786., 2001.],
                      [1., 1717., 1915.],
                      [1., 2198., 2000.],
                      [1., 1362., 1993.],
                      [1., 1694., 2004.],
                      [1., 2090., 1973.],
                      [1., 1774., 1931.],
                      [1., 1077., 1939.]])
    lambda_param = 0.01
    sigma_squared = 1000
    map_coef = calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared)
    ml_coef = calculate_map_coefficients(aug_x, output_y, 0, 0)
    output_y = np.array([208500, 181500, 223500,
                         140000, 250000, 143000,
                         307000, 200000, 129900,
                         118000])
    aug_x = np.array([[1., 1710., 2003.],
                      [1., 1262., 1976.],
                      [1., 1786., 2001.],
                      [1., 1717., 1915.],
                      [1., 2198., 2000.],
                      [1., 1362., 1993.],
                      [1., 1694., 2004.],
                      [1., 2090., 1973.],
                      [1., 1774., 1931.],
                      [1., 1077., 1939.]])

    ml_weights = calculate_map_coefficients(aug_x, output_y, 0, 0)
    sig2 = estimate_data_noise(aug_x, output_y, ml_weights)
    # print(sig2)
