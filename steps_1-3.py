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
    # Does private vs public impact financial aid amount?

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
                                     "vsa_enroll_elsewhere_after6_transfer", "similar", "nicknames", "counted_pct"], axis=1)

# Normalize the continuous variables:
# test on one column first
student_count_normalized = MinMaxScaler().fit_transform(college_drop[['student_count']])
print(student_count_normalized[:10])
college_drop.student_count.plot.density()
pd.DataFrame(student_count_normalized).plot.density()
#apply to all numeric_cols
# select_dtypes('number') finds all int and float columns
numeric_cols = list(college_drop.select_dtypes('number'))
# apply min-max scaling to all numeric columns
college_drop[numeric_cols] = MinMaxScaler().fit_transform(college_drop[numeric_cols])


# Create target variable if needed:
# visualize the aid_value distribution
college_drop.boxplot(column='aid_value', vert=False, grid=False)
# show summary statistics
print(college_drop.aid_value.describe())
#create a binary target variable: aid_f
# 1 = High aid (rating > 0.219), 0 = not high aid (rating <= 0.219)
college_drop['aid_f'] = pd.cut(college_drop.aid_value,
                                bins=[-1, 0.219, 1],
                                labels=[0, 1])
# pd.cut() bins continuous values into discrete categories
# bins=[-1, 0.219, 1] creates two bins: (-1, 0.219] and (0.219, 1]
# labels=[0, 1] assigns 0 to first bin (low aid), 1 to second bin (high aid)
college_drop.info()  # See the new aid_f column


# Calculate the prevalence of the target variable:
prevalence = (college_drop.aid_f.value_counts()[1] /
              len(college_drop.aid_f))
# value_counts()[1] gets count of '1' values (high aid)
# Divide by total count to get proportion
print(f"Baseline/Prevalence: {prevalence:.2%}")
# This is the baseline so models should beat this accuracy


# Create the necessary data partitions (Train,Tune,Test):
#drop features and unrelated columns again before splitting
college_clean = college_drop.drop(['city','state', 'basic', 
                                  'awards_per_value', 'awards_per_state_value', 
                                  'awards_per_natl_value','exp_award_value',
                                  'exp_award_state_value','exp_award_natl_value',
                                    'exp_award_percentile','fte_percentile',
                                    'med_sat_percentile','endow_percentile',
                                    'grad_100_percentile','grad_150_percentile',
                                    'pell_percentile','retain_percentile',
                                      'ft_fac_percentile'], axis=1)
# drop data leakage 
college_clean = college_clean.drop(['aid_value','aid_percentile'], axis=1)
# still too much missing data so drop
college_clean = college_clean.drop(["med_sat_value","endow_value"], axis=1)
#fix aid_f from category to int
college_clean["aid_f"] = college_clean["aid_f"].astype(int)
#drop the one row in y
college_clean = college_clean.dropna(subset=["aid_f"])
#start splitting data
# First split: Separate training data from the rest
train, leftover = train_test_split(
    college_clean,
    train_size=.6,
    stratify=college_clean.aid_f
)
# stratify=college_clean.aid_f ensures class proportions are preserved
# This reduces sampling error and gives more reliable results
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Leftover set shape: {leftover.shape}")
# Second split: Split remaining data into tuning and test sets (50/50)
tune, test = train_test_split(
    leftover,
    train_size=.5,
    stratify=leftover.aid_f
)
print(f"Tune set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")
#check prevalence in each set to confirm stratification worked
print("\nTraining prevalence:")
print((train.aid_f.value_counts(normalize=True) * 100).round(2))
print("\nTuning prevalence:")
print((tune.aid_f.value_counts(normalize=True) * 100).round(2))
print("\nTest prevalence:")
print((test.aid_f.value_counts(normalize=True) * 100).round(2))



# %% 
job.info()
# %% STEP TWO ____________________________________________________________________________________________

#JOB DATA SET
# Write a generic question that this dataset could address:
    # Does your grades in school affect your salary?

    #***adjusted question to be "How do grades in school affect whether you get a job?
    #I explain why I changed it down below in the code

# What is a independent Business Metric for your problem?
    # The average salary of a graduate with a certain grade in school

# Data preparation:

# Correct variable type/class as needed:
# make the str variables with limited distinct values into categorical variables
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
        "degree_t", "workex", "specialisation", "status"]
job[cols] = job[cols].astype('category')
#make salary int instead of float
job['salary'] = job['salary'].astype("Int64")


# One-hot encoding factor variables:
# get_dummies() creates new binary columns for each category level
job_encoded = pd.get_dummies(job, columns=cols)
job_encoded.info()

#I have a realziation here of the connection between teh missing values in salary
#and the status of being place or not. To make this a little easier on myself right now 
#I am going to change my question to be "How do grades in school affect whether you get a job?
#instead of "Do grades affect salary?"

# Normalize the continuous variables:
# test on one column first
ssc_p_normalized = MinMaxScaler().fit_transform(job_encoded[['ssc_p']])
print(ssc_p_normalized[:10])
job_encoded.ssc_p.plot.density()
pd.DataFrame(ssc_p_normalized).plot.density()
#apply to all numeric_cols
# select_dtypes('number') finds all int and float columns
numeric_cols = list(job_encoded.select_dtypes('number'))
# apply min-max scaling to all numeric columns
job_encoded[numeric_cols] = MinMaxScaler().fit_transform(job_encoded[numeric_cols])


# Drop unneeded variables:
#dropping unique identifiers, columns with lots of missing values, and columns not needed for analysis
job_drop = job_encoded.drop(["sl_no", "salary", "gender_F", 
                             "gender_M", "ssc_b_Central", "ssc_b_Others",
                             "hsc_b_Central", "hsc_b_Others", "hsc_s_Arts",
                             "hsc_s_Commerce","hsc_s_Science",
                              "degree_t_Comm&Mgmt", "degree_t_Others",
                               "degree_t_Sci&Tech", "workex_No", "workex_Yes",
                                "specialisation_Mkt&Fin", "specialisation_Mkt&HR" ], axis=1)

#also want to rename to get rid of the space
job_drop.rename(columns={'status_Not Placed': 'status_Not_Placed'}, inplace=True)

job_drop.info()

# Create target variable if needed:
# work backwards to make placed the target variable
# status_Placed = True means they got placed (1), False means not placed (0)
job_drop['placed'] = job_drop['status_Placed'].astype(int)
# drop the one-hot encoded status columns
job_clean = job_drop.drop(['status_Not_Placed', 'status_Placed'], axis=1)
job_clean.info()

# Calculate the prevalence of the target variable:
prevalence = (job_clean.placed.value_counts()[1] / len(job_clean.placed))
print(f"Baseline/Prevalence: {prevalence:.2%}")
#68.84% models should beat this accuracy to be useful

#Create the necessary data partitions (Train,Tune,Test)
#start splitting data
# First split: Separate training data from the rest
train, leftover = train_test_split(
    job_clean,
    train_size=.6,
    stratify=job_clean.placed
)
# stratify=job_clean.TARGET_NAME ensures class proportions are preserved
# This reduces sampling error and gives more reliable results
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Leftover set shape: {leftover.shape}")
# Second split: Split remaining data into tuning and test sets (50/50)
tune, test = train_test_split(
    leftover,
    train_size=.5,
    stratify=leftover.placed
)
print(f"Tune set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")
print("\nTraining prevalence:")
print((train.placed.value_counts(normalize=True) * 100).round(2))
print("\nTuning prevalence:")
print((tune.placed.value_counts(normalize=True) * 100).round(2))
print("\nTest prevalence:")
print((test.placed.value_counts(normalize=True) * 100).round(2))

# %%
# %% STEP THREE ____________________________________________________________________________________________

#College data set:
#Can data address your problem?:
    # Yes, it has info on financial aid amounts and 
    # many other characteristics of colleges that could
    # be used to predict whether a student receives high financial aid or not
#What areas/items are you worried about?:
    # Theres still alot of missing data in some columns like SAT scores and endowment
    # There may be confounding variables that have impacts on aid but arnt in the dataset


#Job data set:
#Can data address your problem?:
    # Yes, it has info on students' grades in school and whether
    #  they got placed in a job or not
#What areas/items are you worried about?:
    # There may be confounding variables that have impacts on placement but 
        #arnt in the dataset like family connections or legacy status
    # The dictionary said that the curriculum and grading standards can 
        # differ across school boards in India, so even though I put it on same scale,
        # the actually content and difficulty of exams could be different
