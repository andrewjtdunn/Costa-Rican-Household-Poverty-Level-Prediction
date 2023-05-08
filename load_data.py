# Load and clean data

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE

SEED = 2023


def load_train_data():
    """
    Loads, cleans, and imputes new variables in Kaggle data

    Input: None

    Returns:
        df (dataframe), composed of X_train and y_train
        X_valid (dataframe)
        y_valid (dataframe)
    """
    # Load data
    df = pd.read_csv("Kaggle_download/train.csv")

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
    df.loc[:, "hh_child_woman_ratio"] = df.loc[:, "r4t1"] / df.loc[:, "r4m3"]

    # child/adult ratio in household
    # children defined as under 12
    # adults defined as 12 and over
    df.loc[:, "hh_child_adult_ratio"] = df.loc[:, "r4t1"] / df.loc[:, "r4t2"]

    # child/woman ratio in household
    # children defined as under 19
    # women defined as 12 and over
    # THIS IS A DATA QUALITY ISSUE -- CATS AREN'T MUTUALLY EXCLUSIVE
    df.loc[:, "hh_child_woman_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "r4m2"]

    # child/adult ratio in household
    # children defined as under 19
    # adults defined as 19 and over
    df.loc[:, "hh_child_adult_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "hogar_adul"]

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

    features_to_include = [
        x
        for x in var_desc.keys()
        if x
        not in [
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

    # Split into test and train
    ###########################################################################
    X_train, X_valid, y_train, y_valid = train_test_split(
        df.drop(columns="Target"),
        df.loc[:, ["Target"]],
        test_size=0.2,
        random_state = SEED,
    )

    # merge the train sets back together
    df = X_train.merge(y_train, left_index=True, right_index=True)

    return df, X_valid, y_valid


# Generate randomly oversampled data
def gen_oversample_date():
    '''
    UPDATE DOC STRING
    '''

    df, X_valid, y_valid = load_train_data()
    X = df.iloc[:, :-1]
    y = df.loc[:, 'Target']

    ros = RandomOverSampler(random_state = SEED)
    train_X_resampled, train_y_resampled = ros.fit_resample(X, y)

    return train_X_resampled, train_y_resampled


# Generate SMOTE data
def gen_SMOTE_data():
    '''
    UPDATE DOC STRING
    '''
    df, X_valid, y_valid = load_train_data()
    X = df.iloc[:, :-1]
    y = df.loc[:, 'Target']

    sm = SVMSMOTE(random_state = SEED )
    X_smote, y_smote = sm.fit_resample(X, y)

    return X_smote, y_smote
