# Helper function to evaluate ML model classifications
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score

def evaluate_classification(y_pred, y_true, l=[1,2,3,4], cm=False, return_vals=False):
    '''
    COMPLETE DOC STRING
    '''
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average=None)
    print("Accuracy:", accuracy.round(2))
    print("F1 Score:", f1.round(2))
    print("Recall:", 'Label 1:', recall[0].round(2), 'Label 2:', recall[1].round(2), 
          'Label 3:', recall[2].round(2), 'Label 4:', recall[3].round(2))
    if cm is True:
        cm = confusion_matrix(y_true, y_pred, labels=l)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=l)
        disp.plot()
    
    if return_vals:
        return accuracy, f1, recall