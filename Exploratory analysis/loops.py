import os
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

from evaluate_classification import evaluate_classification
from sklearn.feature_selection import VarianceThreshold

def loop_model(model, df, train_indices, valid_indices, scaler=None, 
               oversample=None, var_thresh=False):
    """
    This function runs an already k-fold cross validated dataset through a loop
        of each fold through a provided ML model, with optionality to add 
        processing steps. It stores the results of each pass as returned from
        evaluate classification in a dictionary called results, which is returned

    Inputs:
    model (SKLearn Model Object): Any pre-instantiated sklearn model object 
        that has methods fit and transform
    df (Pandas DataFrame): df with both features and target
    train_indices (dictionary): A dictionary with keys referring to an 
        individual fold (from k-fold) and values referring to the indexes to 
        include as training data for that pass
    valid_indices (dictionary): A dictionary with keys referring to an 
        individual fold (from k-fold) and values referring to the indexes to 
        include as validation data for that pass
    scaler (SKLearn Scaling Object, optional): Any pre-instantiated sklearn 
        scaler that has the transform method
    oversample (function, optional): Any pre-existing function (two target
        functions are in load_data) which can be passed in. the function MUST 
        take in a df object and return X and y oversampled
    var_thresh (Boolean): Boolean flag for using a variance threshold to remove
        features with low variance

    Outputs:
    results (dictionary): A dictionary with key being the pass through the 
        dataset and values being the dictionary returned from evaluate 
        classification with the performance of this pass of the model
    """

    # Handling Variance Threshold
    if var_thresh:
        sel = VarianceThreshold(threshold=.16)
        X = df.drop(columns="Target").copy()
        y = df.loc[:, ["Target"]].copy()
        
        X_selected = sel.fit_transform(X)
        # This creates a mask with the columns we're trying to keep
        selected_features = sel.get_support()
        sel_df = pd.DataFrame(X_selected, columns=X.columns[selected_features])

        # The variance threshold object returns an ndarray which removed our index
        sel_df.index = y.index
        sel_df.loc[:,'Target'] = y
        df = sel_df
    
    # Instantiate the results dictionary
    results = {}

    for key in train_indices.keys():

        # If oversampling call provided function, otherwise split by index    
        if oversample:
            df_train = df.iloc[train_indices[key],:]
            X_train, y_train = oversample(df_train)
        else:
            X_train = df.drop(columns="Target").iloc[train_indices[key],:]
            y_train = df.loc[:,['Target']].iloc[train_indices[key],:]
        
        X_valid = df.drop(columns="Target").iloc[valid_indices[key],:]
        y_valid = df.loc[:,['Target']].iloc[valid_indices[key],:]
        
        # If provided scaler, rescale X on both sides
        if scaler:
            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
        
        # Fit our model, then predict it and evaluate performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        results[key] = evaluate_classification(y_pred, y_valid, cm = True, return_vals=True)

    
    return results