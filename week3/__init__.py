import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_posterior_p():
    # GRADED
    # Given the values in the tables above, calculate the posterior for:

    # "If you know a user watches Youtube every day,
    # what is the probability that they are under 35?"

    # Assign float (posterior probability between 0 and 1) to ans1

    # YOUR ANSWER BELOW
    # Comments added by Rushikesh
    # posterior = ( likelihood * prior )/norm_marginal
    # Exa P(E/H) =( P(H/E)     * P(H)  ) /P(E)
    # In THis case P(B/A) = P(A/B) * P(B) /P(A)
    # Likelihood = P(A=1/P(B<35)
    # Prior P(B<35)
    # Marginal P(A=1 )
    # How to calculate norm_marginal
    # P(A=1) = P(A=1/B<=35) * P(B<=35) +
    #         P(A=1/35 < B < 65 ) * P(35 < B < 65 ) +
    #         P(A=1/B >= 65 ) * P(B >= 65 )
    _likelihood = 0.9
    _prior = 0.25
    norm_marginal = (.9 * .25) + (.5 * .45) + (.1 * .3)
    return (_likelihood * _prior) / norm_marginal


# GRADED
# Code a function called `calc_posterior`
# ACCEPT three inputs
# Two floats: the likelihood and the prior
# One list of tuples, where each tuple has two values corresponding to:
# # ( P(Bn) , P(A|Bn) )
# # # Assume the list of tuples accounts for all potential values of B
# # # And that those values of B are all mutually exclusive.
# The list of tuples allows for the calculation of normalization constant.
# RETURN a float corresponding to the posterior probability
# YOUR ANSWER BELOW
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
    marginal_norm = sum([event[0] * event[1] for event in norm_list])
    return (likelihood * prior) / marginal_norm


# GRADED
# Build a function called "x_preprocess"
# ACCEPT one input, a numpy array
#  Array may be one or two dimensions

# If input is two dimensional, make sure there are more rows than columns
#  Then prepend a column of ones for intercept term
# If input is one-dimensional, prepend a one

# RETURN a numpy array, prepared as described above,
# which is now ready for matrix multiplication with regression weights

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

    if input_x.ndim == 1:
        return np.insert(input_x, 0, 1, axis=0)
    elif input_x.ndim == 2:
        rows = len(input_x[:, 0])
        col = len(input_x[0, :])
        if rows < col:
            input_x = np.transpose(input_x)
        ones_col = np.ones(len(input_x[:, 0]))
        return np.insert(input_x, 0, ones_col, axis=1)

    # alternate ways
    # ndim = input_x.ndim
    # if ndim == 2:
    #     rows = len(input_x[:, 0])
    #     col = len(input_x[0, :])
    #     if rows < col:
    #         input_x = np.transpose(input_x)
    #     ones_col = np.ones(len(input_x[:, 0]))
    #     input_x = np.insert(input_x, 0, ones_col, axis=1)
    # if ndim == 1:
    #     input_x = np.insert(input_x, 0, 1, axis=0)
    # return input_x


# GRADED
# Build a function called `calculate_map_coefficients`

# ACCEPT four inputs:
# Two numpy arrays; an X-matrix and y-vector
# Two positive numbers, a lambda parameter, and value for sigma^2

# RETURN a 1-d numpy vector of weights.

# ASSUME your x-matrix has been preprocessed:
# observations are in rows, features in columns, and a column of 1's prepended.

# Use the above equation to calculate the MAP weights.
# # This will involve creating the lambda matrix.
# # The MAP weights are equal to the Ridge Regression weights

# NB: `.shape`, `np.matmul`, `np.linalg.inv`,
# `np.ones`, `np.identity` and `np.transpose` will be valuable.

# If either the "sigma_squared" or "lambda_param" are equal to 0, the return will be
# equivalent to ordinary least squares.

# YOUR ANSWER BELOW

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


# GRADED
# Code a function called `estimate_data_noise`

# ACCEPT three inputs, all numpy arrays
# One matrix corresponding to the augmented x-matrix
# Two vectors, one of the y-target, and one of ML weights.

# RETURN the empirical data noise estimate: sigma^2. Calculated with equation given above.

# NB: "n" is the number of observations in X (rows)
# "d" is the number of features in aug_x (columns)

# YOUR ANSWER BELOW

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
    diff = []
    for element in range(len(output_y)):
        print(element)
        diff.append((output_y[element] - aug_x[element] @ weights) ** 2)
    return sum(diff) / (n - d)


# GRADED
# Code a function called "calc_post_cov_mtx"
# ACCEPT three inputs:
# One numpy array for the augmented x-matrix
# Two floats for sigma-squared and a lambda_param

# Calculate the covariance matrix of the posterior (capital sigma), via equation given above.
# RETURN that matrix.

# YOUR ANSWER BELOW


def calc_post_cov_mtx(aug_x, sigma_squared, lambda_param):
    """
    Calculate the covariance of the posterior for Bayesian parameters

    Positional arguments:
        aug_x -- matrix of training input data; preprocessed
        sigma_squared -- estimation of sigma^2
        lambda_param -- lambda parameter that controls how heavily
        to penalize large weight values

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

        ml_weights = calculate_map_coefficients(aug_x, output_y,0,0)

        sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights)

        print(calc_post_cov_mtx(aug_x, sigma_squared, lambda_param))
        # --> [[ 9.99999874e+01 -1.95016334e-02 -2.48082095e-02]
               [-1.95016334e-02  6.28700339e+01 -3.85675510e+01]
               [-2.48082095e-02 -3.85675510e+01  5.10719826e+01]]

    Assumptions:
        -- training_input_y is a vector whose length is the same as the
        number of rows in training_x
        -- lambda_param has a value greater than 0

    """
    return np.linalg.inv(lambda_param * np.identity(aug_x.shape[1]) + (1 / sigma_squared * (aug_x.T @ aug_x)))


# GRADED
# Code a function called "predict"
# ACCEPT four inputs, three numpy arrays, and one number:
# A 1-dimensional array corresponding to an augmented_x vector.
# A vector corresponding to the MAP weights, or "mu"
# A square matrix for the "big_sigma" term
# A positive number for the "sigma_squared" term

# Using the above equations

# RETURN mu_0 and sigma_squared_0 - a point estimate and variance
# for the prediction for x.

# YOUR ANSWER BELOW

def predict(aug_x, weights, big_sig, sigma_squared):
    """
    Calculate point estimates and uncertainty for new values of x

    Positional Arguments:
        aug_x -- augmented matrix of observations for predictions
        weights -- MAP weights calculated from Bayesian LR
        big_sig -- The posterior covarience matrix, from Bayesian LR
        sigma_squared -- The observed uncertainty in Bayesian LR

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

        ml_weights = calculate_map_coefficients(aug_x, output_y,0,0)

        sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights)

        map_weights = calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared)

        big_sig = calc_post_cov_mtx(aug_x, sigma_squared, lambda_param)

        to_pred2 = np.array([1,1700,1980])

        print(predict(to_pred2, map_weights, big_sig, sigma_squared))
        #-->(158741.6306608729, 1593503867.9060116)

    """
    mu_0 = aug_x.T@weights
    sigma_squared_0 = sigma_squared+(aug_x.T@big_sig)@aug_x
    return mu_0, sigma_squared_0


if __name__ == '__main__':
    calc_posterior_p()
    likelihood = .8
    prior = .3
    norm_list = [(.25, .9), (.5, .5), (.25, .2)]
    # print(calc_posterior(likelihood, prior, norm_list))
    input1 = np.array([[2, 3, 6, 9], [4, 5, 7, 10]])
    # input2 = np.array([2, 3, 6])
    # input3 = np.array([[2, 4], [3, 5], [6, 7], [9, 10]])
    # print(x_preprocess(input1))
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
    # print(calculate_map_coefficients(aug_x, output_y, lambda_param, sigma_squared))
    # print(calculate_map_coefficients(aug_x, output_y, 0, 0))
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
    # sig2 = estimate_data_noise(aug_x, output_y, ml_weights)

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
    ml_weights = calculate_map_coefficients(aug_x, output_y, 0, 0)
    sigma_squared = estimate_data_noise(aug_x, output_y, ml_weights)
    print(calc_post_cov_mtx(aug_x, sigma_squared, lambda_param))
