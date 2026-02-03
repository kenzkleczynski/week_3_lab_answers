# %% Imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests


# _______________________________ COLLEGE DATA SET _____________________________________________________
# %% STEP ONE ____________________________________________________________________________________________

# load college data
college_url = "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
college = pd.read_csv(college_url)
print(college.info())

# Observations:
    # Alot of missing values like in sat columns, endownment columns, vsa columns, nicknames)
    # Some uncessesayr co,umns like webURL, nickname, index, unitID
    # Some columns with a lot of unique values like institution name, city, address

# Problems that could be sovled with this data:
    #Help a student choose a college based on their prefrences
    #Help a student predict colleges they have a change of getting into based on their stats
    # A college could use it to compare themselves to others, like in the same state/size/type

# Questions:
    # Does awards impact or correlate with gradutaion rates?
    # Does your SAT score limit your chances of getting into certain colleges by states or private vs public?
    # Does being in a college of certain state imact your finacial aid amount?



# %% STEP TWO ____________________________________________________________________________________________

# Write a generic question that this dataset could address:
    # Does state and private vs public impact financial aid amount?

# What is a independent Business Metric for your problem?
    # Average financial aid amount per student

# Data preparation:

# Correct variable type/class as needed:
# Level as 2 distinct values and control as 3 distinct values so make to categorical
cols = ["level", "control"]
college[cols] = college[cols].astype('category')
# Year is float but should be int
college["vsa_year"] = college["vsa_year"].astype('Int64')
# Make hbsu and flagship boolean
college["hbcu"] = college["hbcu"].astype('boolean')
college["flagship"] = college["flagship"].astype('boolean')


# Collapse factor levels as needed: *


# One-hot encoding factor variables:
college_encoded = pd.get_dummies(college, columns=cols)
# get_dummies() creates new binary columns for each category level
# Original categorical columns are removed, replaced by indicator columns
# Check the result
college_encoded.info()



# Normalize the continuous variables:


# Drop unneeded variables:
# Create target variable if needed:
# Calculate the prevalence of the target variable:
# Create the necessary data partitions (Train,Tune,Test):

# %%



# %%
