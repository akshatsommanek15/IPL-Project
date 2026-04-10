import os
import sys
from dataclasses import dataclass
from sklearn.metrics import classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.preprocessor import Preprocessor
from src.utils import evaluate_models

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, groups_train):
        try:

            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier":CatBoostClassifier(),
                "LGBMClassifier":LGBMClassifier()
            }

            preprocessor = Preprocessor().get_preprocessor()

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test,
                models, preprocessor, groups_train
            )

            best_model_name = max(model_report, key=lambda x: model_report[x]['cv_score'])
            best_model = model_report[best_model_name]["model"]

            y_pred = best_model.predict(X_test)

            print(f"\nBest Model: {best_model_name}")
            print(f"Accuracy: {model_report[best_model_name]['test_accuracy']:.4f}")

            print("Classification Report:\n")
            print(classification_report(
                y_test, y_pred,
                target_names=["Defending Team Wins", "Chasing Team Wins"]
            ))

            save_object(self.config.trained_model_file_path, best_model)

        except Exception as e:
            raise CustomException(e, sys)