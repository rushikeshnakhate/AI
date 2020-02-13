import unittest
import numpy as np
from knn import *


class TestCase(unittest.TestCase):

    def test_find_euclidean_distance(self):
        p1 = (5, 6, 7, 8, 9, 10)
        p2 = (1, 2, 3, 4, 5, 6)
        self.assertEqual(find_euclidean_distance(p1, p2), 9.797958971132712)

    def test_all_distances(self):
        data_set = read_train_data()
        test_point = data_set.iloc[50, :]
        actual = [0.0, 2.7970187358249854, 2.922792670143521, 2.966555149052483, 3.033982453218797]
        np.testing.assert_allclose(all_distances(test_point, data_set)[:5], actual)

    def test_labels_of_smallest(self):
        numeric = np.array([7, 6, 5, 4, 3, 2, 1])
        labels = np.array(["a", "a", "b", "b", "b", "a", "a"])
        n = 6
        expected = np.array(['a', 'a', 'b', 'b', 'b', 'a'])
        actual = labels_of_smallest(numeric, labels, n)
        np.testing.assert_array_equal(actual, expected)

    def test_no_null_in_train_data(self):
        df = read_train_data()
        self.assertEqual(df.isnull().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
