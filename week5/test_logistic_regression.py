import unittest
import pandas as pd
from __init__ import *


class TestLogisticRegression(unittest.TestCase):
    def test_tr_path(self):
        df = get_tr()
        df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], inplace=True, axis=1)
        df = df.loc[df.Embarked.notnull(), :]
        y_target = df.Survived
        df.drop(['Survived'], axis=1, inplace=True)
        print(df.head())
        to_dummy = ['Sex']
        new = pd.get_dummies(df, prefix=to_dummy, columns=to_dummy, drop_first=True)
        print(new.head())


if __name__ == '__main__':
    unittest.main()
