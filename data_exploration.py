import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


import torch
from torch.utils.data import DataLoader, TensorDataset


data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def basic_data_exploration(df: pd.DataFrame):
    '''
    for a given DF it checks:
    columns
    missing data
    values of non-numeric columns
    basic statistics of dataset
    '''

    dataset = df.copy()

    ## basic info of dataset
    print("\nSUMMARY OF DATASET\n")
    print(dataset.info())
    print("\n", "="*100, "\n")

    ## further checking for null or NaN values
    # print("NaN or Null \n")
    print("\nNaN or Null")
    for c in dataset.columns:
        print("")
        print(" All values/row for ", c, " = ", len(dataset[c]))
        print(" NaN count: \n", dataset[c].isna().value_counts())
        # print("Null count: \n", dataset[c].isnull().value_counts())  ### it is the same as .isna() in pandas
        print("")
    print("\n", "="*100, "\n")

    ## non-numeric columns
    non_num_cols = []
    for c in dataset.columns:
        if dataset[c].dtypes == 'object':
            non_num_cols.append(c)
    print("\nThe non-numeric columns are:\n")
    print(non_num_cols)
    print("\n", "="*100, "\n")

    ## values of non-numeric columns
    print("\nVALUES OF NON-NUMERIC COLUMNS")
    for c in non_num_cols:
        print("")
        print(dataset[c].value_counts(), "\n")
        print(" Normalized ", dataset[c].value_counts(normalize=True))
        print("")
    print("\n", "="*100, "\n")


    ## non-missing values (neither empty or NaN) of non-num columns, but zero information values
    ## like empty string: "",  or space string: " ", or tab: "\t"  etc.
    print("\nWHITESPACE VALUES OF NON_NUMERIC COLUMNS")
    for c in non_num_cols:
        print("")
        print("Column: ", c)
        print((dataset[c].str.strip() == "").sum())
        print("")
    print("\n", "=" * 100, "\n")


### First look of the dataset
# basic_data_exploration(data)


def count_values_alert(df: pd.DataFrame, alert_threshold: int = 2):
    '''
    If a column has way too many distinct/unique values it alerts us. The usefulness is twofold:
        1) This is important because we might have columns like, Name, CustomerNimber/CustomerID etc.
           which will be useless for our model to learn from, so we can drop them.
        2) It pinpoints columns like Height, Age, Salary etc. that
            i) we may want to group into categories, e.g. 1.70-1.79 height -> 1.75 or medium height etc.
            ii) these may be non-numerical columns, but have numbers as strings, so we may turn them into
                numerical columns straight away

    After running  basic_data_exploration  we can make an educated guess of what the alert_thrshold should be
    '''
    dataset = df.copy()

    for c in dataset.columns:
        unique_values = dataset[c].unique()
        num_of_unique_vals = dataset[c].nunique(dropna=True)
        if num_of_unique_vals > alert_threshold:
            if dataset[c].dtypes == 'object':
                print("\nNon-Numerical column")
                print(f"Number of unique values for column '{c}' is  {num_of_unique_vals}")
                print(unique_values)
                print("")
            else:
                print("\nNumerical column")
                print(f"Number of unique values for column '{c}' is  {num_of_unique_vals}")
                print(unique_values)
                print("")



# count_values_alert(data)

data2 = data.copy()
data2 = data.drop(columns="customerID")

one_hot_num_cols = ['SeniorCitizen']

non_num_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']

str_num_cols = ["TotalCharges"]

categorial_no_num_cols = [c for c in non_num_cols if c not in str_num_cols]


######## Binary categorial values to numeric values
##### checking the non-numeric binary categorial values
binary_no_num_cols = []
for c in non_num_cols:
    if data2[c].dtypes == "object":
        all_unique_vals = data2[c].unique()
        if data2[c].nunique(dropna=True) == 2:
            # print(c, all_unique_vals)
            binary_no_num_cols.append(c)


rest_categorial_non_num_cols = [c for c in categorial_no_num_cols if c not in binary_no_num_cols]



