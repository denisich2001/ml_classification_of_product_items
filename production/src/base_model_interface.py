from abc import ABC
from sklearn.model_selection import train_test_split
from abc import abstractmethod


class ClassificatorModelInterface(ABC):
    """
    Абстрактный класс для класса модели классификатора:
        * Разбиение на обучающую и тестовую выборки
        * params_tuner - подбор гиперпараметров на кросс-валидации (optuna)
        * Обучение модели
        * Оценка качества
        * Формирование итогового предсказание
    """
    def __init__(self) -> None:
        self.trainset_features = None
        self.trainset_target = None
        self.classifier_model = None
        super().__init__()

    def trainset_train_test_split(self, test_size: float = 0.3):
        """
        Метод разбиения трейнсета на обучающую и тестовую выборки

        test_size - отношение размера тестовой выборки (по умолчанию = 0.3)
        """
        features_train, features_test, target_train, target_test = train_test_split(
            self.trainset_features,
            self.trainset_target,
            test_size
        )
        return features_train, features_test, target_train, target_test

    def classificator_predict(self, products_for_classification):
        """
        Метод выполняющий классификацию для переданных ему данных
        """
        # todo доделать
        predicted_classes = self.classifier_model.predict(products_for_classification)
        return predicted_classes

