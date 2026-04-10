import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            # Drop unwanted columns
            train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
            test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

            # Replace inf with NaN
            train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            target_column = "result"
            group_column = "match_id"

            # Split features and target
            X_train = train_df.drop(columns=[target_column, group_column])
            y_train = train_df[target_column]
            groups_train = train_df[group_column]

            X_test = test_df.drop(columns=[target_column, group_column])
            y_test = test_df[target_column]

            return X_train, y_train, X_test, y_test, groups_train

        except Exception as e:
            raise CustomException(e, sys)