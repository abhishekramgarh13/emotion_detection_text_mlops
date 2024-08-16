import numpy as np
import pandas as pd 
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging

logger = logging.getLogger("model evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#importing test data
def load_test_data(file_path):
    try:
        test_data = pd.read_csv(file_path)
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values
        logger.info("test data loaded successfully.")
        return X_test,y_test
    except Exception as e:
        logger.error(f"error in loading test data {e}.")
        raise e

#loading model
def evaluate_model(X_test,y_test):
    try:
        clf = pickle.load(open('./models/model.pkl','rb'))

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        with open('reports/metrics.json','w') as file:
            json.dump(metrics_dict,file, indent=4)
    except Exception as e:
        logger.error(f'error in evaluating model {e}.')
        raise e
    
def main():
    X_test,y_test = load_test_data('./data/features/test_tfidf.csv')
    evaluate_model(X_test,y_test)

if __name__=="__main__":
    main()