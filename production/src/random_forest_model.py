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
        accuracy = super().prepare_model()
        logger.info('Обучение модели закончено.')
        return accuracy

    @staticmethod
    def get_params(trial):
        # todo допилить
        max_depth = trial.suggest_int('max_depth', 5, 150, step=5)
        n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=10)
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
        rf_model = RandomForestClassifier(**params)
        logger.debug(f'Проверим гиперпараметры модели: {params}')
        cross_validation_mean_accuracy = super().cross_validation(
            rf_model, train_x, train_y
        )
        return cross_validation_mean_accuracy
    
    def fit_model(self, train_x, train_y):
        logger.info('Обучаем итоговую версию модели')
        if self.model_best_params:
            self.classifier_model = RandomForestClassifier(**(self.model_best_params))
        else:
            self.classifier_model = RandomForestClassifier()
        super().fit_model(train_x, train_y)
        logger.info('Итоговая версия модели обучена!')
