import pandas as pd
import numpy as np

# Accuracy, f1, recall
def select_best(result_dict, selector, label=None):
    """
    DOC STRING TK
    """
    if label:
        scores = [result_dict[x][selector][label-1] for x in result_dict.keys()]
    else:
        scores = [result_dict[x][selector] for x in result_dict.keys()]
    
    return np.argmax(scores)


def average_outcome(result_dict):
    """
    DOC STRING TK
    """
    
    avg_dict = {}
    avg_dict['accuracy'] = np.mean([result_dict[x]['accuracy'] for x in result_dict.keys()])
    avg_dict['f1'] = np.mean([result_dict[x]['f1'] for x in result_dict.keys()])
    avg_dict['recall_1'] = np.mean([result_dict[x]['recall'][0] for x in result_dict.keys()])
    avg_dict['recall_2'] = np.mean([result_dict[x]['recall'][1] for x in result_dict.keys()])
    avg_dict['recall_3'] = np.mean([result_dict[x]['recall'][2] for x in result_dict.keys()])
    avg_dict['recall_4'] = np.mean([result_dict[x]['recall'][3] for x in result_dict.keys()])

    return avg_dict


