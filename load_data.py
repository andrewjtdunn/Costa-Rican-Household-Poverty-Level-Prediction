# Load and clean data

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE
from exploratory_analysis import loops
import os

SEED = 2023


def load_train_data(filepath="Kaggle_download/train.csv", seed=SEED):
    """
    Loads, cleans, and imputes new variables in Kaggle data

    Input: 
        file (csv): optional, default is the train data
        seed (int): optional seed

    Returns:
        df (dataframe), composed of X_train and y_train
        X_valid (dataframe)
        y_valid (dataframe)
    """
    # Load data
    df = pd.read_csv(filepath)

    # Clean a couple data fields
    ###########################################################################
    # see here for info: https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/discussion/61751

    # edjefe
    df.loc[df.loc[:, "edjefe"] == "yes", "edjefe"] = 1
    df.loc[df.loc[:, "edjefe"] == "no", "edjefe"] = 0
    df["edjefe"] = df["edjefe"].astype(str).astype(int)

    # edjefa
    df.loc[df.loc[:, "edjefa"] == "yes", "edjefa"] = 1
    df.loc[df.loc[:, "edjefa"] == "no", "edjefa"] = 0
    df["edjefa"] = df["edjefa"].astype(str).astype(int)

    # ASSUME DEPENDENCY HAS THE SAME MISCODING
    # https://www.kaggle.com/competitions/costa-rican-household-povertyx-prediction/discussion/73055
    df.loc[df.loc[:, "dependency"] == "yes", "dependency"] = 1
    df.loc[df.loc[:, "dependency"] == "no", "dependency"] = 0
    df["dependency"] = df["dependency"].astype(str).astype(float)

    # Fix NAs for number of tablets owned
    df.loc[:, "v18q1"] = df.loc[:, "v18q1"].fillna(0)

    # Create new individual-level variables base on lit review
    ###########################################################################

    # highest level of education in household
    def get_max_education_level(row):
        education_levels = [
            row["instlevel1"],
            row["instlevel2"],
            row["instlevel3"],
            row["instlevel4"],
            row["instlevel5"],
            row["instlevel6"],
            row["instlevel7"],
            row["instlevel8"],
            row["instlevel9"],
        ]
        return max(education_levels)

    # Create a new column in the DataFrame representing the highest education level in a household
    df["max_education_level"] = df.apply(get_max_education_level, axis=1)

    # if there is a marriage in the household
    df.loc[:, "hh_has_marriage"] = (
        df.loc[:, "estadocivil3"].groupby(df.loc[:, "idhogar"]).transform("max")
    )

    # max age in household
    df.loc[:, "hh_max_age"] = (
        df.loc[:, "age"].groupby(df.loc[:, "idhogar"]).transform("max")
    )

    # sex ratio in household
    # #male/#female
    df.loc[:, "hh_sex_ratio"] = df.loc[:, "r4h3"] / df.loc[:, "r4m3"]

    # child/woman ratio in household
    # children defined as under 12
    # women defined as 12 and over
    df.loc[:, "hh_child_woman_ratio_12"] = df.loc[:, "r4t1"] / df.loc[:, "r4m3"]

    # child/adult ratio in household
    # children defined as under 12
    # adults defined as 12 and over
    df.loc[:, "hh_child_adult_ratio_12"] = df.loc[:, "r4t1"] / df.loc[:, "r4t2"]

    # child/woman ratio in household
    # children defined as under 19
    # women defined as 12 and over
    # THIS IS A DATA QUALITY ISSUE -- CATS AREN'T MUTUALLY EXCLUSIVE
    df.loc[:, "hh_child_woman_ratio_19"] = df.loc[:, "hogar_nin"] / df.loc[:, "r4m2"]

    # child/adult ratio in household
    # children defined as under 19
    # adults defined as 19 and over
    df.loc[:, "hh_child_adult_ratio_19"] = df.loc[:, "hogar_nin"] / df.loc[:, "hogar_adul"]

    # Reshape the data to be at household level rather than individual level
    ###########################################################################

    # pick the head of the household
    df.loc[df.loc[:, "parentesco1"] == 1, "hh_head"] = 1

    # create temp vars to determine if household head exists and max age in household
    df.loc[:, "hh_head_exists"] = df.groupby([df.loc[:, "idhogar"]])[
        "hh_head"
    ].transform(max)

    # in instances where there isn't a head of household, pick the oldest male
    df.loc[
        (
            (df.loc[:, "hh_head_exists"] == 0)
            & (df.loc[:, "age"] == df.loc[:, "hh_max_age"])
            & (df.loc[:, "male"] == 1)
        ),
        "hh_head",
    ] = 1

    # update the temp hh head flag var
    df.loc[:, "hh_head_exists"] = df.groupby([df.loc[:, "idhogar"]])[
        "hh_head"
    ].transform(max)

    # in instances where there isn't an oldest male, pick the oldest
    df.loc[
        (
            (df.loc[:, "hh_head_exists"] == 0)
            & (df.loc[:, "age"] == df.loc[:, "hh_max_age"])
        ),
        "hh_head",
    ] = 1

    # collapse the data
    df = df.loc[df.loc[:, "hh_head"] == 1]

    # drop the temp var and other household head vars
    df = df.drop(columns=["hh_head_exists", "parentesco1", "hh_head"])

    # Create household-level variables
    ###########################################################################

    with open("var_descriptions.json", "r") as f:
        # Load JSON data as a dictionary
        var_desc = json.load(f)

    features_to_include = [x for x in var_desc.keys() if x not in [
            "Id",
            "idhogar",
            "dependency",
            "rez_esc",
            "hh_head",
            "parentesco1",
            "hh_head_exists",
        ]
    ]
    df_subset = df[features_to_include]

    # impute mean rent values while suppressing error message
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=Warning)
        imp_mean = IterativeImputer(random_state=0, n_nearest_features=5)
        imp_mean.fit(df_subset)
        mean_subset = imp_mean.transform(df_subset)

    # replace 0s    
    df.loc[:, "v2a1"] = mean_subset[:, 0]
    df.loc[df.loc[:, "v2a1"] < 0, "v2a1"] = 0

    # define logged value of v2a1, it provides a better distribution
    df["v2a1_log"] = np.log1p(df["v2a1"])

    # Clean up NAs and inf values
    cols_to_drop = ['Id', 'idhogar', 'rez_esc']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)

    train_indices, valid_indices = implement_kfold(df)

    return df, train_indices, valid_indices

def implement_kfold(df, n_splits=5, shuffle=True, random_state=SEED):
    """
    This helper function implements stratified k-fold cross validation. 
        Primarily called within the load_data function but can be called 
        independently

    Inputs:
    df (DataFrame): a dataframe with features and a target column
    n_splits (int, optional): k, the number of splits to make in the dataframe
    shuffle (Bool, optional): Whether to shuffle each class's samples before 
        splitting into batches.
    random_state (int, optional): the random seed to set for replicability

    Outputs:
    train_indices (dictionary): A dictionary with keys referring to an 
        individual fold (from k-fold) and values referring to the indexes to 
        include as training data for that pass
    valid_indices (dictionary): A dictionary with keys referring to an 
        individual fold (from k-fold) and values referring to the indexes to 
        include as validation data for that pass
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = skf.split(df.drop(columns="Target"), df.loc[:, ["Target"]])

    train_indices = {}
    valid_indices = {}

    for i, (train_index, valid_index) in enumerate(indices):
        train_indices[i] = train_index
        valid_indices[i] = valid_index

    return train_indices, valid_indices

def gen_oversample_data(df, seed = SEED):
    '''
    Generate resampled dataframes.

    Inputs:
        df (dataframe): data and labels 
        seed (int): optional seed
    
    Returns:
        train_X_resampled (dataframe): the resampled data
        train_y_resampled (dataframe): the resampled labels

    '''

    X = df.iloc[:, :-1]
    y = df.loc[:, 'Target']

    ros = RandomOverSampler(random_state = seed)
    train_X_resampled, train_y_resampled = ros.fit_resample(X, y)

    return train_X_resampled, train_y_resampled

def gen_SMOTE_data(df, seed = SEED):
    '''
    Generate SMOTE dataframes.

    Inputs:
        df (dataframe): data and labels 
        seed (int): optional seed
    
    Returns:
        X_smote (dataframe): the resampled data
        y_smote (dataframe): the resampled labels
    '''
    X = df.drop(columns='Target')
    y = df.loc[:, 'Target']

    sm = SVMSMOTE(random_state = seed)
    X_smote, y_smote = sm.fit_resample(X, y)

    return X_smote, y_smote


def two_step(model, df, train_indices, valid_indices, oversample=None, var_thresh=False):
    print("Classification for 4")
    loops.loop_model(model,df,train_indices,valid_indices,oversample=None,var_thresh=False)
    not_pred= df.loc[df.loc[:,'Target'].isin([1,2,3]),:]
    train_indices_not, valid_indices_not = implement_kfold(not_pred, n_splits=5, shuffle=True, random_state=SEED)
    print("Classification for 1,2,3")
    loops.loop_model(model,not_pred,train_indices_not, valid_indices_not ,oversample=oversample ,var_thresh=var_thresh)
