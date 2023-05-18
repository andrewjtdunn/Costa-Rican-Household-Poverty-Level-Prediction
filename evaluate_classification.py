# Helper function to evaluate ML model classifications
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score

def evaluate_classification(y_pred, y_true, l=[1,2,3,4], cm=False, return_vals=False):
    '''
    COMPLETE DOC STRING
    '''
    eval_dict = {}
    eval_dict['accuracy'] = accuracy_score(y_true, y_pred)
    eval_dict['f1'] = f1_score(y_true, y_pred, average="weighted")
    eval_dict['Macro f1'] = f1_score(y_true, y_pred, average="macro")
    eval_dict['recall'] = recall_score(y_true, y_pred, average=None)
    eval_dict['precision'] = recall_score(y_true, y_pred, average=None)
    print("Accuracy:", eval_dict['accuracy'].round(2))
    print("F1 Score:", eval_dict['f1'].round(2))
    print("Macro F1 Score:", eval_dict['Macro f1'].round(2))
    print("Recall:", 'Label 1:', eval_dict['recall'][0].round(2), 'Label 2:', 
          eval_dict['recall'][1].round(2))
    if len(eval_dict['recall']) == 4:
          print('      Label 3:', eval_dict['recall'][2].round(2))
          print('      Label 4:', eval_dict['recall'][3].round(2))
    if cm is True:
        cm = confusion_matrix(y_true, y_pred, labels=l)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=l)
        disp.plot()
    
    if return_vals:
        return eval_dict