import unittest
from random_classifier import *
import matplotlib.pyplot as plt


class TestRandomClassifier(unittest.TestCase):
    def test_default_payment(self):
        df = get_df()
        defaulter = df[df['default payment next month'] == 0]
        self.assertEqual((defaulter['default payment next month'].count()), 23364)

    def test_get_not_default_payment(self):
        df = get_df()
        non_default_payment = df[df['default payment next month'] == 1]
        self.assertEqual(non_default_payment['default payment next month'].count(), 6636)

    def test_eduction_data(self):
        df = get_df()
        # print(df['EDUCATION'].value_counts())
        # print(df['MARRIAGE'].value_counts())
        # print(df['SEX'].value_counts())
        # for i in [-2, -1, 0, 1, 2]:
        #     print(df[df['PAY_0'] == i][['PAY_0', 'BILL_AMT1', 'PAY_AMT1']].head(10), "\n")


if __name__ == '__main__':
    unittest.main()
