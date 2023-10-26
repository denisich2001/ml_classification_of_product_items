import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from production.src.base_model_interface import ClassificatorModelInterface


class RandomForestModel(ClassificatorModelInterface):
    """
    Класс модели случайного леса
    """
    def __init__(self, trainset_features: pd.DataFrame, trainset_target: pd.DataFrame) -> None:
        super().__init__(trainset_features, trainset_target, RandomForestClassifier())

    def prepare_model(self):
        """
        Основной метод подготовки модели для классификации:
        * Разбиение на train/test
        * Обучение модели
        """
        logger.info('Начинаем обучение модели случайного леса.')
        super().prepare_model()
        logger.info('Обучение модели закончено.')
