import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from production.config import UseCrossValidation
from production.src.base_model_interface import ClassificatorModelInterface


class RandomForestModel(ClassificatorModelInterface):
    """
    Класс модели случайного леса
    """
    def __init__(self, product_table_for_train: pd.DataFrame) -> None:
        super().__init__(RandomForestClassifier(), product_table_for_train)

    def prepare_model(self):
        """
        Основной метод подготовки модели для классификации:
        * Разбиение на train/test
        * Обучение модели
        """
        logger.info('Начинаем обучение модели случайного леса.')
        accuracy = super().prepare_model()
        logger.info('Обучение модели закончено.')
        return accuracy

    @staticmethod
    def get_params(trial):
        max_depth = trial.suggest_int('max_depth', 5, 150, step=5)
        n_estimators = trial.suggest_int('n_estimators', 10, 300, step=5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 6, 14, step=2)
        return {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'min_samples_leaf': min_samples_leaf
        }

    def objective(self, trial, train_x, train_y):
        model_params = self.get_params(trial)
        rf_model = RandomForestClassifier(**model_params)
        logger.debug(f'Проверим гиперпараметры модели: {model_params}')
        if UseCrossValidation:
            cross_validation_mean_accuracy = super().cross_validation(
                rf_model, train_x, train_y
            )
            logger.debug(f'Точность на текущих гиперпараметрах: {cross_validation_mean_accuracy}')
            return cross_validation_mean_accuracy
        else:
            train_x, val_x, train_y, val_y = self.trainset_train_test_split(train_x,train_y)
            rf_model.fit(train_x, train_y)
            predicted_val_y = rf_model.predict(val_x)
            accuracy = accuracy_score(val_y, predicted_val_y)
            logger.debug(f'Точность на текущих гиперпараметрах: {accuracy}')
            return accuracy

