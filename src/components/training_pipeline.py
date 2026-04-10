from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, y_train, X_test, y_test, groups = transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(X_train, y_train, X_test, y_test, groups)