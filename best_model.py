# This file contains our best model after extensive testing

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
import ml_model_testing.analyze_k
import load_data
import loops


def best_model(df):
    """
    Defines the best model and trains it on all training data

    Input (dataframe): the cleaned training data

    Returns (model object): log_reg, the trained model
    """
    sel = VarianceThreshold(threshold=0.16)
    X = df.drop(columns="Target").copy()
    y = df.loc[:, ["Target"]].copy()

    X_selected = sel.fit_transform(X)
    # This creates a mask with the columns we're trying to keep
    selected_features = sel.get_support()
    sel_df = pd.DataFrame(X_selected, columns=X.columns[selected_features])

    # The variance threshold object returns an ndarray which removed our index
    sel_df.index = y.index
    sel_df.loc[:, "Target"] = y
    df = sel_df

    # Apply SMOTE oversampling
    X_smote, y_smote = load_data.gen_SMOTE_data(df=df)

    log_reg = LogisticRegression(solver="liblinear", penalty="l2")

    # Fit model on the SMOTE data
    log_reg.fit(X_smote, y_smote)

    return log_reg
