# %% Imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests



# %% STEP ONE ____________________________________________________________________________________________

# load college data
college_url = "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
college = pd.read_csv(college_url)
print(college.info())

# Observations:
    # Alot of missing values (especially in sat columns, endownment columns, vsa columns, nicknames)
    # 
    


# Problems that could be sovled with this data:
    #

# Questions:
    # 



# %% load job data
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


