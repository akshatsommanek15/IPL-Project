import sys
import optuna
import os
import dill
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, preprocessor, groups_train):
    try:
        report = {}

        for model_name, model in models.items():

            def objective(trial):

                if model_name == "Random Forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 400),
                        "max_depth": trial.suggest_int("max_depth", 5, 12),
                        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
                        "bootstrap": True,
                        "max_samples": trial.suggest_float("max_samples", 0.7, 0.9),
                        "random_state": 42,
                        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
                    }

        

                elif model_name == "XGBClassifier":
                    params = {
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), # Added log
                        "max_depth": trial.suggest_int("max_depth", 3, 8),
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400), # Fixed to int
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                        "use_label_encoder": False,
                        "eval_metric": "logloss",
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                        "gamma": trial.suggest_float("gamma",0.1,2.0, log=True),
                        "random_state": 42
                    }

                elif model_name == "CatBoost Classifier":
                        params = {
                            "depth": trial.suggest_int("depth", 4, 8),
                            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
                            "iterations": trial.suggest_int("iterations", 800, 2500),
                            "random_strength": trial.suggest_float("random_strength", 1e-8, 10, log=True),
                            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "MVS"]),
                            "grow_policy": "SymmetricTree",
                            "eval_metric": "Logloss",
                            "random_state": 42,
                            "verbose": 0,
                            "allow_writing_files": False
                        }

                elif model_name == "LGBMClassifier":
                    params = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "verbosity": -1,
                        "boosting_type": "gbdt",
                        "random_state": 42,
                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                        "max_depth": trial.suggest_int("max_depth", 5, 11),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
                        "path_smooth": trial.suggest_float("path_smooth", 0, 10)
                    }
                                

                # Create fresh model with params
                model_instance = model.__class__(**params)

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model_instance)
                ])

                # Group-aware CV
                gkf = GroupKFold(n_splits=5)

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=gkf,
                    groups=groups_train,
                    scoring="accuracy",
                    n_jobs=-1
                )

                return np.mean(scores)

            study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=300)

            # Best params
            best_params = study.best_params

            # Train final pipeline
            raw_best_model = model.__class__(**best_params)
            calibrated_model=CalibratedClassifierCV(
                estimator=raw_best_model,
                method="isotonic",
                cv=5
            )

            final_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", calibrated_model)
            ])

            final_pipeline.fit(X_train, y_train)

            y_pred = final_pipeline.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            report[model_name] = {
                "model": final_pipeline,
                "cv_score": study.best_value,
                "test_accuracy": test_acc
            }

        return report
    except Exception as e:
        raise CustomException(e,sys)

