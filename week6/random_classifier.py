import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_df():
    return pd.read_excel(r'D:\AI\week6\data\default of credit card clients.xls', header=1)


### Define function for creating histograms
def pay_hist(df, cols, ymax):
    plt.figure(figsize=(10, 7))  # define fig size

    for index, col in enumerate(cols):  # For each column passed to function
        plt.subplot(2, 3, index + 1)  # plot on new subplot
        plt.ylim(ymax=ymax)  # standardize ymax
        plt.hist(df[col])  # create hist
        plt.title(col)  # title with column names
    plt.tight_layout();  # make sure titles don't overlap
    df[pay_amt_cols].boxplot()
    plt.show(block=True)


def rename(df):
    df['SEX'] = df['SEX'] - 1;
    df.rename(columns={'SEX': 'FEMALE', "default payment next month": "default"}, inplace=True)

    # for col, pre in zip(["EDUCATION", "MARRIAGE"], ["EDU", "MAR"]):
    #     df = pd.concat([
    #         df.drop(col, axis="columns"), pd.get_dummies(df[col], prefix=pre, drop_first=True)],
    #         axis='columns'
    #     )
    #     # print(df.head())
    print(df["EDUCATION"])
    print(pd.get_dummies(df["EDUCATION"], prefix="EDU", drop_first=True))


if __name__ == '__main__':
    df = get_df()
    # pay_cols = ["PAY_" + str(n) for n in range(2, 7)]
    # pay_amt_cols = ['PAY_AMT' + str(n) for n in range(1, 7)]
    # bill_amt_cols = ['BILL_AMT' + str(n) for n in range(1, 7)]
    # pay_hist(df, pay_cols, 20000)
    # df[pay_amt_cols].boxplot()
    # df_no_0_pay_amt_1 = df[df["PAY_AMT1"] != 0]
    # print(df_no_0_pay_amt_1["PAY_AMT1"].hist())
    #
    # log_pay_amt1 = np.log10(df_no_0_pay_amt_1["PAY_AMT1"])
    # plt.hist(log_pay_amt1)
    # pay_hist(df, bill_amt_cols, 23000)
    rename(df)
