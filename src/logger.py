import logging
import os
from datetime import datetime

LOG_FILE_NAME = f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log'
LOGS_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

os.makedirs(LOGS_FILE_PATH, exist_ok= True)

LOGS_PATH = os.path.join(LOGS_FILE_PATH, LOG_FILE_NAME)


logging.basicConfig(
    filename= LOGS_PATH,
    format= "[%(asctime)s] %(lineno)s %(name)s - %(levelname)s -%(message)s",
    level= logging.INFO

)