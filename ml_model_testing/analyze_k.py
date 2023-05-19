import pandas as pd
import numpy as np

# Accuracy, f1, recall
def select_best(result_dict, selector, label=None):
    """
    This function takes a dictionary where each value is a dictionary returned 
        from our evaluate classification function. It returns the index of the 
        model / model pass that performs best on the desired selector

    Inputs:
    result_dict (dictionary): a dictionary with each value being a dictionary 
        returned by evaluate classification
    selector (immutable): a key which exists in result dict
    label (int, optional): used for recall, tells the function which label we 
        want to find the best recall score
    
    Outputs:
    Int-- the index of the model with the best performance on the selected metric
    """

    if label:
        scores = [result_dict[x][selector][label-1] for x in result_dict.keys()]
    else:
        scores = [result_dict[x][selector] for x in result_dict.keys()]
    
    return np.argmax(scores)


def average_outcome(result_dict):
    """
    This function takes a result dictionary with results from evaluate 
        classification and returns a dictionary with the average performance in 
        the inputted dictionary across the metrics that evaluate classification
        considers. While this function would run on any inputted dictionary with
        the following keys, it is designed to average performance across k-folds.
    
    Inputs:
    result_dict (dictionary): a dictionary with each value being a dictionary 
        returned by evaluate classification
    
    Outputs:
    avg_dict (dictionary): a dictionary with keys being the entire set of 
        performance metrics from evaluate classification and values being the 
        mean performance on each metric from the input dictionary
    """
    
    avg_dict = {}
    avg_dict['accuracy'] = np.mean([result_dict[x]['accuracy'] for x in result_dict.keys()])
    avg_dict['f1'] = np.mean([result_dict[x]['f1'] for x in result_dict.keys()])
    avg_dict['Macro f1'] = np.mean([result_dict[x]['Macro f1'] for x in result_dict.keys()])

    num_cats = len(result_dict[0]['recall'])
    if num_cats == 2:
        avg_dict['recall_non_pov'] = np.mean([result_dict[x]['recall'][0] for x in result_dict.keys()])
        avg_dict['recall_pov'] = np.mean([result_dict[x]['recall'][1] for x in result_dict.keys()])
        avg_dict['precision_non_pov'] = np.mean([result_dict[x]['precision'][0] for x in result_dict.keys()])
        avg_dict['precision_pov'] = np.mean([result_dict[x]['precision'][1] for x in result_dict.keys()])
    for i in range(num_cats):
        avg_dict[f'recall_{i+1}'] = np.mean([result_dict[x]['recall'][i] for x in result_dict.keys()])
        avg_dict[f'precision_{i+1}'] = np.mean([result_dict[x]['precision'][i] for x in result_dict.keys()])

    return avg_dict


