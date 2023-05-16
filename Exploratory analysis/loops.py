import os
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

from evaluate_classification import evaluate_classification
from sklearn.feature_selection import VarianceThreshold

def loop_model(model, df, train_indices, valid_indices, scaler=None, oversample=None, var_thresh=False):
    """
    DOC STRING TK
    """

    if var_thresh:
        sel = VarianceThreshold(threshold=.16)
        X = df.drop(columns="Target").copy()
        y = df.loc[:, ["Target"]].copy()
        X_selected = sel.fit_transform(X)
        selected_features = sel.get_support()
        sel_df = pd.DataFrame(X_selected, columns=X.columns[selected_features])
        sel_df.index = y.index
        sel_df.loc[:,'Target'] = y

        df = sel_df.copy()
    results = {}
    for key in train_indices.keys():
        if oversample:
            df_train = df.iloc[train_indices[key],:]
            X_train, y_train = oversample(df_train)
        else:
            X_train = df.drop(columns="Target").iloc[train_indices[key],:]
            y_train = df.loc[:,['Target']].iloc[train_indices[key],:]
        
        X_valid = df.drop(columns="Target").iloc[valid_indices[key],:]
        y_valid = df.loc[:,['Target']].iloc[valid_indices[key],:]
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        results[key] = evaluate_classification(y_pred, y_valid, cm = True, return_vals=True)

        return results