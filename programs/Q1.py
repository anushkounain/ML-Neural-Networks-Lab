#Program 1: Perform Data preprocessing on health care data 
#The program uses pandas and sklearn. Please install these packages
#pip install pandas
#pip install -U scikit-learn
#pip install matplotlib

import pandas as pd
#from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as py

# Load health dataset into panda's data frame
data = pd.read_csv(r"DATASET FILE LOCATION")

# 1. Count missing values per column
missing_counts_before = data.isnull().sum()
print('Missing values Before cleaning: ',missing_counts_before[missing_counts_before > 0])

# Drop columns with missing values 
data_cleaned = data.dropna()

missing_counts_after = data_cleaned.isnull().sum()
print('Missing values After Cleaning: ',missing_counts_after[missing_counts_after > 0])

num_cols = data_cleaned.select_dtypes(include=["float64", "int64"]).columns

print('Number of columns: ',num_cols)

print('Summary Statistics: ')
print(data_cleaned.describe())

