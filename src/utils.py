import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Fitting the model

            model.fit(X_train, y_train)

            # Prediction using model

            y_pred = model.predict(X_test)

            # Model accuracy

            score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)