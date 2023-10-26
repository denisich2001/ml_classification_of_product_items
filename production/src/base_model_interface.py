from abc import ABC
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class ClassificatorModelInterface(ABC):
    """
    Абстрактный класс для класса модели классификатора:
        * Разбиение на обучающую и тестовую выборки
        * params_tuner - подбор гиперпараметров на кросс-валидации (optuna)
        * Обучение модели
        * Оценка качества
        * Формирование итогового предсказание
    """
    def __init__(self, trainset_features: pd.DataFrame, trainset_target: pd.DataFrame, classifier_model) -> None:
        self.trainset_features = trainset_features
        self.trainset_target = trainset_target
        self.classifier_model = classifier_model
    # TODO ОЦЕНКА ТОЧНОСТИ
    # TODO ПОДБОР ГИПЕРПАРАМЕТРОВ

    def prepare_model(self):
        """
        Основной метод, реализующий весь процесс взаимодействия с моделью.
        * Разбиение на train/test
        * Обучение модели
        """
        train_x, test_x, train_y, test_y = self.trainset_train_test_split()
        self.fit_model(train_x, test_x, train_y, test_y)

    def fit_model(self, train_x, test_x, train_y, test_y):
        """
        Основной метод обучения модели классификации:
        * Тьюнинг гиперпараметров
        * Обучение модели
        * Оценка качества
        """
        # TODO ДОДЕЛАТЬ
        self.classifier_model.fit(train_x, train_y)

    def trainset_train_test_split(self, test_size_value: float = 0.3):
        """
        Метод разбиения трейнсета на обучающую и тестовую выборки

        test_size - отношение размера тестовой выборки (по умолчанию = 0.3)
        """
        logger.info(f'Разбиваем выборку на train/test. Размер теста - {test_size_value}')
        train_x, test_x, train_y, test_y = train_test_split(
            self.trainset_features,
            self.trainset_target,
            test_size=test_size_value
        )
        return train_x, test_x, train_y, test_y

    def predict_classes(self, products_for_classification):
        """
        Метод выполняющий классификацию для переданных ему данных
        """
        predicted_classes = self.classifier_model.predict(products_for_classification)
        predicted_classes_df = pd.DataFrame(predicted_classes)
        return predicted_classes_df

