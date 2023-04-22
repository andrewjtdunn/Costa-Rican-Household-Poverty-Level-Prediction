# Load and clean data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_train_data():
    """
    to do: update doc string
    """
    # Load data
    df = pd.read_csv("Kaggle_download/train.csv")

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
    df.loc[:, "hh_child_woman_ratio"] = df.loc[:, "r4t1"] / df.loc[:, "r4m3"]

    # child/adult ratio in household
    # children defined as under 12
    df.loc[:, "child_adult_ratio"] = df.loc[:, "r4t1"] / df.loc[:, "r4t2"]

    # child/woman ratio in household
    # children defined as under 19
    df.loc[:, "hh_child_woman_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "r4m3"]

    # child/adult ratio in household
    # children defined as under 19
    df.loc[:, "child_adult_ratio"] = df.loc[:, "hogar_nin"] / df.loc[:, "hogar_adul"]

    # Categorize household and individual variables
    ###########################################################################
    hh_vars = [
        "v2a1",
        "hacdor",
        "rooms",
        "hacapo",
        "v14a",
        "refrig",
        "v18q",
        "v18q1",
        "r4h1",
        "r4h2",
        "r4h3",
        "r4m1",
        "r4m2",
        "r4m3",
        "r4t1",
        "r4t2",
        "r4t3",
        "tamhog",
        "tamviv",
        "hhsize",
        "paredblolad",
        "paredzocalo",
        "paredpreb",
        "pareddes",
        "paredmad",
        "paredzinc",
        "paredfibras",
        "paredother",
        "pisomoscer",
        "pisocemento",
        "pisoother",
        "pisonatur",
        "pisonotiene",
        "pisomadera",
        "techozinc",
        "techoentrepiso",
        "techocane",
        "techootro",
        "cielorazo",
        "abastaguadentro",
        "abastaguafuera",
        "abastaguano",
        "public",
        "planpri",
        "noelec",
        "coopele",
        "sanitario1",
        "sanitario2",
        "sanitario3",
        "sanitario5",
        "sanitario6",
        "energcocinar1",
        "energcocinar2",
        "energcocinar3",
        "energcocinar4",
        "elimbasu1",
        "elimbasu2",
        "elimbasu3",
        "elimbasu4",
        "elimbasu5",
        "elimbasu6",
        "epared1",
        "epared2",
        "epared3",
        "etecho1",
        "etecho2",
        "etecho3",
        "eviv1",
        "eviv2",
        "eviv3",
        "idhogar",
        "hogar_nin",
        "hogar_adul",
        "hogar_mayor",
        "hogar_total",
        "dependency",
        "edjefe",
        "edjefa",
        "meaneduc",
        "bedrooms",
        "overcrowding",
        "tipovivi1",
        "tipovivi2",
        "tipovivi3",
        "tipovivi4",
        "tipovivi5",
        "computer",
        "television",
        "mobilephone",
        "qmobilephone",
        "lugar1",
        "lugar2",
        "lugar3",
        "lugar4",
        "lugar5",
        "lugar6",
        "area1",
        "area2",
        "SQBhogar_total",
        "SQBedjefe",
        "SQBhogar_nin",
        "SQBovercrowding",
        "SQBdependency",
        "SQBmeaned",
        "Target",
    ]

    ind_vars = [
        "escolari",
        "rez_esc",
        "dis",
        "male",
        "female",
        "estadocivil1",
        "estadocivil2",
        "estadocivil3",
        "estadocivil4",
        "estadocivil5",
        "estadocivil6",
        "estadocivil7",
        "parentesco1",
        "parentesco2",
        "parentesco3",
        "parentesco4",
        "parentesco5",
        "parentesco6",
        "parentesco7",
        "parentesco8",
        "parentesco9",
        "parentesco10",
        "parentesco11",
        "parentesco12",
        "instlevel1",
        "instlevel2",
        "instlevel3",
        "instlevel4",
        "instlevel5",
        "instlevel6",
        "instlevel7",
        "instlevel8",
        "instlevel9",
        "age",
        "SQBescolari",
        "SQBage",
        "agesq",
    ]

    # Split into test and train
    ###########################################################################
    # last_col = df.pop("Target")
    # df = df.insert(0, last_col.name, last_col)
    # df = df.reindex(columns = [col for col in df.columns if col != 'Target'] + ['Target'])
    X_train, X_valid, y_train, y_valid = train_test_split(
        df.drop(columns='Target'), df.loc[:,['Target']], test_size=0.2, random_state=2023)

    return X_train, X_valid, y_train, y_valid
