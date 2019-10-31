import unittest
import numpy as np
from start import *


class TestCase(unittest.TestCase):
    def test_inverse_of_matrix(self):
        arr = np.array([[1, 2], [3, 4]])
        expected = np.array([[-2, 1], [1.5, -.5]])
        actual = inverse_of_matrix(arr)
        np.testing.assert_allclose(expected, actual)

    def test_least_squares_weights(self):
        training_x = np.array([[1710, 1262, 1786,
                                1717, 2198, 1362,
                                1694, 2090, 1774,
                                1077], [
                                   2003, 1976, 2001,
                                   1915, 2000, 1993,
                                   2004, 1973, 1931,
                                   1939
                               ]])
        training_y = np.array([[208500, 181500, 223500,
                                140000, 250000, 143000,
                                307000, 200000, 129900,
                                118000]])
        expected = np.array([[-2.29223802e+06],
                             [5.92536529e+01],
                             1.20780450e+03])
        actual = least_squares_weights(training_x, training_y)
        np.testing.assert_allclose(expected[0], actual[0])
        np.testing.assert_allclose(expected[1], actual[1])
        np.testing.assert_allclose(expected[2], actual[2])


if __name__ == '__main__':
    unittest.main()
