import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch

## these we can deduce from the basic exploration
fixed_binary_dicts = {
    "yes-no": {"Yes": 1, "No": 0},
    "female-male": {"Female": 1, "Male": 0}
}


def to_num_cols(df: pd.DataFrame, binary_lst: list, non_binary_lst: list, str_num_cols: list):
    '''
    After understanding the basic stucture of the data:
        Numerical strings to Numbers (if they exist)
        Encoding Categorial Values of non-numeric columns
            Binary: manually
            Otherwise: one-hot (.get_dummies())
        !!! In each case, NaNs etc, remain NaN !!!
    Returns a new dataframe based on the original, but now with all columns numerical
    '''

    dataset = df.copy()

    for c in binary_lst:
        v_0, v_1 = dataset[c].dropna().unique()
        for d in fixed_binary_dicts:
            if v_0 in fixed_binary_dicts[d]:
                replace_d = fixed_binary_dicts[d]
                dataset[c] = dataset[c].replace(replace_d)
                break

    dataset = pd.get_dummies(dataset, columns=non_binary_lst, drop_first=True, dtype=int)

    for c in str_num_cols:
        dataset[c] = pd.to_numeric(dataset[c], errors='coerce')

    return dataset


def basic_preprocess_data(df: pd.DataFrame):

    dataset = df.copy()

    ## this we deduce useless, from the basic exploration
    dataset2 = dataset.drop(columns="customerID")

    ## define column groups
    ## these we get from the basic exploration
    non_num_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']
    str_num_cols = ["TotalCharges"]
    categorial_no_num_cols = [
        c for c in non_num_cols
        if c not in str_num_cols
    ]

    ## identify binary categorical columns
    binary_no_num_cols = []
    for c in non_num_cols:
        if dataset2[c].dtype == "object":
            if dataset2[c].nunique(dropna=True) == 2:
                binary_no_num_cols.append(c)

    rest_categorial_non_num_cols = [
        c for c in categorial_no_num_cols
        if c not in binary_no_num_cols
    ]


    dataset3 = to_num_cols(
        dataset2,
        binary_no_num_cols,
        non_binary_lst=rest_categorial_non_num_cols,
        str_num_cols=str_num_cols
    )

    return dataset3







def prepare_for_training(df: pd.DataFrame, Y_col: str, pytorch_bool: bool = False, check_imbalance: bool = False, stratify: bool = True, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42):
    dataset = df.copy()

    dataset_prepr = basic_preprocess_data(dataset)

    X = dataset_prepr.drop(columns=Y_col)
    Y = dataset_prepr[Y_col]

    if check_imbalance:
        print(Y.value_counts())
        print(Y.value_counts(normalize=True))


    ## test-train split
    if stratify:
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_size, random_state=42, stratify=Y_train_val)
    else:
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_size, random_state=42)


    ## handle missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.set_output(transform="pandas")
    ## antileakage principle: only fitting on TRAIN data!
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)


    ## standardization:  avoid one-hot encoded colums
    not_one_hot_num_cols = []
    for c in dataset_prepr.columns:
        unique_vals_num = dataset_prepr[c].nunique(dropna=True)
        if unique_vals_num > 2:
            not_one_hot_num_cols.append(c)
    # print(not_one_hot_num_cols)
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train[not_one_hot_num_cols] = scaler.fit_transform(X_train[not_one_hot_num_cols])
    X_val[not_one_hot_num_cols] = scaler.transform(X_val[not_one_hot_num_cols])
    X_test[not_one_hot_num_cols] = scaler.transform(X_test[not_one_hot_num_cols])


    ## prepare for torch Log Regr model
    if pytorch_bool:
        X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
        X_val_torch = torch.tensor(X_val.values, dtype=torch.float32)
        X_test_torch = torch.tensor(X_test.values, dtype=torch.float32)

        Y_train_torch = torch.tensor(Y_train.values, dtype=torch.float32)
        Y_val_torch = torch.tensor(Y_val.values, dtype=torch.float32)
        Y_test_torch = torch.tensor(Y_test.values, dtype=torch.float32)

        return X_train_torch, X_test_torch, X_val_torch, Y_train_torch, Y_val_torch, Y_test_torch

    return X_train, X_test, X_val, Y_train, Y_val, Y_test






