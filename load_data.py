# Load and clean data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_train_data():
    """
    Loads, cleans, and imputes new variables in Kaggle data

    Input: None

    Returns:
        X_train (dataframe)
        X_valid (dataframe)
        y_train (dataframe)
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
    # df.loc[:, "edjefe"] = df.loc[:, "edjefe"].astype(int)
    df["edjefe"] = df["edjefe"].astype(str).astype(int)

    # edjefa
    df.loc[df.loc[:, "edjefa"] == "yes", "edjefa"] = 1
    df.loc[df.loc[:, "edjefa"] == "no", "edjefa"] = 0
    # df.loc[:, "edjefa"] = df.loc[:, "edjefa"].astype(int)
    df["edjefa"] = df["edjefa"].astype(str).astype(int)

    # ASSUME DEPENDENCY HAS THE SAME MISCODING
    # https://www.kaggle.com/competitions/costa-rican-household-povertyx-prediction/discussion/73055
    df.loc[df.loc[:, "dependency"] == "yes", "dependency"] = 1
    df.loc[df.loc[:, "dependency"] == "no", "dependency"] = 0
    df["dependency"] = df["dependency"].astype(str).astype(float)

    # Fix NAs for number of tablets owned
    df.loc[:, "v18q1"] = df.loc[:, "v18q1"].fillna(0)

    # Create new variables base on lit review
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
    df.loc[:, "child_adult_ratio"] = df.loc[:, "r4t1"] / df.loc[:, "r4t2"]

    # child/woman ratio in household
    # children defined as under 19
    # women defined as 12 and over
    # THIS IS A DATA QUALITY ISSUE -- CATS AREN'T MUTUALLY EXCLUSIVE
    df.loc[:, "hh_child_woman_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "r4m2"]

    # child/adult ratio in household
    # children defined as under 19
    # adults defined as 19 and over
    df.loc[:, "child_adult_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "hogar_adul"]

    # define logged value of v2a1, it provides a better distribution
    df.loc[:, "v2a1_log"] = np.log1p(
        df.loc[:, "v2a1"].fillna(np.mean(df.loc[:, "v2a1"]))
    )

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

    # Split into test and train
    ###########################################################################
    # last_col = df.pop("Target")
    # df = df.insert(0, last_col.name, last_col)
    # df = df.reindex(columns = [col for col in df.columns if col != 'Target'] + ['Target'])
    X_train, X_valid, y_train, y_valid = train_test_split(
        df.drop(columns="Target"),
        df.loc[:, ["Target"]],
        test_size=0.2,
        random_state=2023,
    )

    return X_train, X_valid, y_train, y_valid
