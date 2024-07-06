import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import evaluate_models


# Initializing model trainer config
@dataclass
class ModelTrainerConfig:
    trained_model_obj_file_path = os.path.join('artifacts', 'model.pkl')

# Initializing model transformation class
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:

            logging.info("Dividing train and test data")

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'SVR': SVR(),
                "RF": RandomForestRegressor(),
                'DT': DecisionTreeRegressor(),
                'ElasticNet': ElasticNet(),
                'KN': KNeighborsRegressor()}

            logging.info("Model training started")

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            logging.info(f"Model Report: {model_report}")

            best_modal_score = max(sorted(model_report.values()))

            best_modal_name = list(model_report.keys())[list(model_report.values()).index(best_modal_score)]

            best_modal = models[best_modal_name]

            logging.info(f"Best Model Found: Model Name: {best_modal_name}, accuracy: {best_modal_score}")

            save_object(obj = best_modal, file_path= self.model_trainer_config.trained_model_obj_file_path )

        except Exception as e:
            raise CustomException(e,sys)        

