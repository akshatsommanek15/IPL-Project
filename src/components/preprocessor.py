import sys
import os
from dataclasses import dataclass
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from src.exception import CustomException

@dataclass
class PreprocessorConfig:
    pass

class Preprocessor:
    def __init__(self):
        self.config = PreprocessorConfig()

    def get_preprocessor(self):
        try:
            numeric_columns = ['runs_target','balls_left','runs_left','wickets_left','current_run_rate','toss_advantage','required_run_rate','is_tail_batting','over_left','wickets_fallen','wicket_pressure','momentum','runs_per_wicket','balls_per_wicket']
            ohe_columns =['batting_team','bowling_team','toss_decision','phase']

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='mean'))
            ])

            ohe_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("onehot", OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ])

            
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_columns),
                ("ohe", ohe_pipeline, ohe_columns),
            ])

            return preprocessor.set_output(transform="pandas")

        except Exception as e:
            raise CustomException(e, sys)