import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation



if __name__ == '__main__':
    # Data Ingestion Pipeline
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    # Data Transformation Pipeline
    data_tranformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_tranformation_obj.initiate_data_transformation(train_data_path, test_data_path)
