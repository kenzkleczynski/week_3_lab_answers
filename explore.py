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
    # Help a student choose a college based on their prefrences
    # Help a student predict colleges they have a change of getting into based on their stats
    # A college could use it to compare themselves to others, like in the same state/size/type

# Questions:
    # Does awards impact or correlate with gradutaion rates?
    # Does your SAT score limit your chances of getting into certain colleges by states or private vs public?
    # Does being in a college of certain state imact your finacial aid amount?

#load job data
job_url = "https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv"
job = pd.read_csv(job_url)
print(job.info())


# Observations:
    # Currency is in Rupee
    # All of the _p (percentages) are just the grade on that "final" exam on a 100 point scale
    # India has differnt school boards/pathways for education that have different curriculumns and rigor

# Problems that could be sovled with this data:
    # Predicting salaries based on what type of degree you have
    # Predicting salary based on grades you got in school
    # Accessing how you school board (in india) affects your salary/employment chances

# Questions:
    # Are there gender pay gaps?
    # Does the type of degree you have affect your salary?
    # Does your grades in school affect your salary?



# %% STEP TWO ____________________________________________________________________________________________

#COLLEGE DATA SET
# Write a generic question that this dataset could address:
    # Does state and private vs public impact financial aid amount?

# What is a independent Business Metric for your problem?
    # Average financial aid amount per student

# Data preparation:

# Correct variable type/class as needed:
# Level as 2 distinct values and control as 3 distinct values so make to categorical.
# hbcu and flagship are binary variables so make categorical too
cols = ["level", "control", "hbcu", "flagship" ]
college[cols] = college[cols].astype('category')
# Year is float but should be int
college["vsa_year"] = college["vsa_year"].astype('Int64')


# One-hot encoding factor variables:
# get_dummies() creates new binary columns for each category level
college_encoded = pd.get_dummies(college, columns=cols)


# Drop unneeded variables:
#dropping unique identifiers, columns with lots of missing values, and columns not needed for analysis
college_drop = college_encoded.drop(["index", "unitid", "chronname", "long_x", "lat_y",
                                     "site", "vsa_year", "vsa_grad_after4_first", 
                                     "vsa_grad_elsewhere_after4_first", "vsa_enroll_after4_first",
                                     "vsa_enroll_elsewhere_after4_first", "vsa_grad_after6_first",
                                     "vsa_grad_elsewhere_after6_first", "vsa_enroll_after6_first",
                                     "vsa_enroll_elsewhere_after6_first", "vsa_grad_after4_transfer",
                                     "vsa_grad_elsewhere_after4_transfer", "vsa_enroll_after4_transfer",
                                     "vsa_enroll_elsewhere_after4_transfer", "vsa_grad_after6_transfer",
                                     "vsa_grad_elsewhere_after6_transfer", "vsa_enroll_after6_transfer", 
                                     "vsa_enroll_elsewhere_after6_transfer", "similar", "nicknames"], axis=1)

college_drop.info()

# Normalize the continuous variables:


# Drop unneeded variables:
# Create target variable if needed:
# Calculate the prevalence of the target variable:
# Create the necessary data partitions (Train,Tune,Test):

# %%
college_drop.select_dtypes(include=["int64", "float64"]).columns
num_cols = ["awards_per_value", "awards_per_state_value","awards_per_natl_value",
            "exp_award_value", "exp_award_per_value", "exp_award_per_state_value", 
            "exp_award_per_natl_value", ""]


# %%
