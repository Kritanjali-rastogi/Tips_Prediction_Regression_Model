import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass


# Initializing Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Initializing Data Transformation Class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        logging.info('Data Transformation Initiated')

        
        try:
            # Segregating numerical and categorical columns
            numerical_columns = ['total_bill', 'size']
            categorical_columns = ['sex', 'smoker', 'day', 'time']

            # Developing pipelines
            logging.info('Data Pipelines Initiated')
            numerical_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy= 'median')),
                                                  ('scaler', StandardScaler())])

            categorical_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy= 'most_frequent')),
                                                    ('encoder', OneHotEncoder())])

            preprocessor = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_columns),
                                              ("categorical_pipeline", categorical_pipeline, categorical_columns)])
            
            
            return preprocessor
            logging.info('Preprocessor oject made')
        
        except Exception as e:
            logging.info('Error in preprocessing')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_data_path, test_data_path):

        # Reading data after data ingestion

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data')
            logging.info(f'Train data head: {train_df.head().to_string()}')
            logging.info(f'Test data head: {test_df.head().to_string()}')

            # Segreggating independend and dependent features

            target_column_name = 'tip'
            drop_features = ['tip']

            input_features_train_df = train_df.drop(columns= drop_features, axis=1)
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = train_df.drop(columns=drop_features, axis=1)
            target_features_test_df = train_df[target_column_name]

            logging.info('Input and target features segrregated for both train and test dataset')

            # Applying feature engineering

            logging.info('Feature Engineering started')

            preprocessor_obj = self.get_preprocessor_obj()

            input_features_train_df_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_df_arr = preprocessor_obj.transform(input_features_test_df)

            logging.info('Feature Engineering completed')

            train_arr = np.c_[input_features_train_df_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_df_arr, np.array(target_features_test_df)]

            save_object(obj = preprocessor_obj,
                    file_path= self.data_transformation_config.preprocessor_obj_file_path)
        
            logging.info('Preprocessor created and saved')

            return (train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info('Error in preprocessing')
            raise CustomException(e,sys)



