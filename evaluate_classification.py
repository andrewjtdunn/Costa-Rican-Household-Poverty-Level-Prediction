# Helper function to evaluate ML model classifications
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score

def evaluate_classification(y_pred, y_true, l=[1,2,3,4], cm=False, return_vals=False):
    '''
    COMPLETE DOC STRING
    '''
    eval_dict = {}
    eval_dict['Accuracy'] = accuracy_score(y_true, y_pred)
    eval_dict['Weighted F1'] = f1_score(y_true, y_pred, average="weighted")
    eval_dict['Macro F1'] = f1_score(y_true, y_pred, average="macro")
    eval_dict['Recall'] = recall_score(y_true, y_pred, average=None)
    print("Accuracy:", eval_dict['Accuracy'].round(2))
    print("Weighted F1 Score:", eval_dict['Weighted F1'].round(2))
    print("Macro F1 Score:", eval_dict['Macro F1'].round(2))
    print("Recall:", 'Label 1:', eval_dict['Recall'][0].round(2), 'Label 2:', 
          eval_dict['Recall'][1].round(2), 'Label 3:', eval_dict['Recall'][2].round(2), 
          'Label 4:', eval_dict['Recall'][3].round(2))
    if cm is True:
        cm = confusion_matrix(y_true, y_pred, labels=l)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=l)
        disp.plot()
    
    if return_vals:
        return eval_dict