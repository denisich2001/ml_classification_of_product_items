import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from production.src.handle_trainset import DataHandler
from production.src.base_model_interface import ClassificatorModelInterface


class RandomForestModel(ClassificatorModelInterface):
    """
    Класс модели случайного леса
    """
    def __init__(
            self,
            product_table_for_train: pd.DataFrame,
            input_table_types_dict: dict
    ) -> None:
        super().__init__(
            RandomForestClassifier(),
            product_table_for_train,
            input_table_types_dict
        )

    def prepare_model(self):
        """
        Основной метод подготовки модели для классификации:
        * Разбиение на train/test
        * Обучение модели
        """
        logger.info('Начинаем обучение модели случайного леса.')
        super().prepare_model()
        logger.info('Обучение модели закончено.')

    @staticmethod
    def get_params(trial):
        # todo допилить
        max_depth = trial.suggest_int('max_depth', 1, 13, step=3)
        n_estimators = trial.suggest_int('n_estimators', 50, 150, step=5)
        #min_samples_leaf = trial.suggest_int()
        #min_samples_split = trial.suggest_int()
        return {
            'max_depth': max_depth,
            'n_estimators': n_estimators
            #'max_features': max_features,
            #'min_samples_leaf': min_samples_leaf,
            #'min_samples_split': min_samples_split,
        }

    def objective(self, trial, train_x, train_y):
        """

        """
        # TODO ДОПИЛИТЬ
        params = self.get_params(trial)
        #logger.debug(f'Current params: {params}')
        rf_model = RandomForestClassifier(**params)
        logger.debug(f'Проверим гиперпараметры модели: {params}')
        cross_validation_mean_accuracy = super().cross_validation(
            rf_model, train_x, train_y
        )
        return cross_validation_mean_accuracy
    
    def fit_model(self, train_x, train_y):
        self.classifier_model = RandomForestClassifier(self.model_best_params)
        super().fit_model(train_x, train_y)
