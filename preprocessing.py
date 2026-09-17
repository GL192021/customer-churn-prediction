import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, recall_score, precision_score


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

    # binart_non_num_cols = []
    # numerical_str_columns = []

    # return non_num_cols

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

non_num_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']

str_num_cols = ["TotalCharges"]

categorial_no_num_cols = [c for c in non_num_cols if c not in str_num_cols]
# print(categorial_no_num_cols)

######## Binary categorial values to numeric values
##### checking the non-numeric binary categorial values
binary_no_num_cols = []
for c in non_num_cols:
    if data2[c].dtypes == "object":
        all_unique_vals = data2[c].unique()
        if data2[c].nunique(dropna=True) == 2:
            # print(c, all_unique_vals)
            binary_no_num_cols.append(c)

# yes_no_dict={"No":0, "Yes":1}
# male_fem_dict = {"Male":0, "Female":1}
# dictss = {
#     "yes_no": yes_no_dict,
#     "male_fem": male_fem_dict
# }
# print(dictss.values())

rest_categorial_non_num_cols = [c for c in categorial_no_num_cols if c not in binary_no_num_cols]
# print(rest_categorial_non_num_cols)

def building_custom_and_concistent_binary_dict(df: pd.DataFrame, binary_columns_lst):
    '''
    Returns a dictionary of dictionaries.
    The keys are the binary categorial non-num columns and the respective dictionary is a custom dictionary we build
    to replace the non-numeric values with 0-1.
    For consistency, binary columns for with same values, share the same dictionary.
    '''

    dataset = df.copy()

    dictionaries = {}

    acceptable = [int(0), int(1)]
    for c in binary_columns_lst:
        v0, v1 = dataset[c].dropna().unique()
        print('Build the binary numeric dictionary for ', v0, v1)
        replacment_dict = None
        if dictionaries:
            for diction in dictionaries.values():
                if v0 in diction:
                    replacment_dict = diction
                    print("We have already done that binary conversion to 0-1")
                    print(replacment_dict)
                    break
            if replacment_dict is not None:
                dictionaries[c] = replacment_dict
                continue

        v0_replacment = int(input(f"zero or one for {v0}: "))
        while v0_replacment not in acceptable:
            print("Wrong choice, has to be 0 or 1")
            v0_replacment = int(input(f"zero or one for {v0}: "))
        v1_replacment = int(1-v0_replacment)
        replacment_dict = {v0: v0_replacment, v1:v1_replacment}
        dictionaries[c] = replacment_dict

    return dictionaries


custom_binary_dicts = building_custom_and_concistent_binary_dict(data2, binary_no_num_cols)

def minimum_preprocessing(df: pd.DataFrame, binary_dicts: dict, non_binary_lst: list =None, str_num_cols: list = None):
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


###### BIG QUESTION: What happens if we had missing values, NaNs etc?????
######               What is a good practice? should we ignore/drop them right from the beginning? should we create a
######               third label??
######               Because, doing something like average, or most frequent replacement in a 0-1 columnn does not
######               really make much sense. It mostly feels like adding unecessary noise!
######     For now, it stays as it is because I do not have missing values
######               Maybe it requires to firstly split the dataset in train-test and then handling this
    for c in binary_dicts:
        dictionary = binary_dicts[c]
        dataset[c] = dataset[c].replace(dictionary)

    dataset = pd.get_dummies(dataset, columns=non_binary_lst, drop_first=True, dtype=int)

    for c in str_num_cols:
        dataset[c] = pd.to_numeric(dataset[c], errors='coerce')

    return dataset








# def prepare_for_training(dataset: pd.DataFrame, Y_col: str,     check_imbalance: bool = True,, test_size: float = 0.2, random_state: int = 42):