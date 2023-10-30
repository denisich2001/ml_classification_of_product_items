import numpy as np
import optuna
from abc import ABC
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from production.config import СVSplitsNumber
from production.config import TargetColumnName
from production.config import OptunaTrialsNumber


class ClassificatorModelInterface(ABC):
    """
    Абстрактный класс для класса модели классификатора:
        * Разбиение на обучающую и тестовую выборки
        * params_tuner - подбор гиперпараметров на кросс-валидации (optuna)
        * Обучение модели
        * Оценка качества
        * Формирование итогового предсказания
    """
    def __init__(
            self,
            classifier_model,
            product_table_for_train: pd.DataFrame,
    ) -> None:
        self.trainset_target = product_table_for_train[TargetColumnName]
        self.trainset_features = product_table_for_train.drop(TargetColumnName, axis=1)
        self.classifier_model = classifier_model
        self.model_best_params = None

    def prepare_model(self):
        """
        Основной метод, реализующий весь процесс взаимодействия с моделью.
        """
        train_x, test_x, train_y, test_y = self.trainset_train_test_split(self.trainset_features, self.trainset_target)
        self.params_tuner(train_x, train_y)
        self.fit_model(train_x, train_y)
        return self.check_quality(test_x, test_y)

    def params_tuner(self, train_x, train_y):
        """
        Метод обучения модели:
        * Тьюнинг гиперпараметров с кросс-валидацией
        * Определение лучших гиперпараметров и обучение модели на них
        """
        logger.info('Подбираем оптимальные гиперпараметры для модели')
        study = optuna.create_study(direction="maximize")
        logger.debug(f'Размеры тренировочной выборки для кросс-валидации: {train_x.shape}')
        objective_func = lambda trial: self.objective(trial, train_x, train_y)
        study.optimize(objective_func, n_trials=OptunaTrialsNumber)
        # Сохраняем лучшие гиперпараметры
        self.model_best_params = study.best_params
        self.classifier_model.set_params(**self.model_best_params)
        logger.debug(f'Оптимальные значения гиперпараметров:\n{self.model_best_params}')
        logger.debug(f'Среднее значение accuracy после проведения кросс-валидации:\n{study.best_value}')

    def objective(self, trial, train_x, test_x):
        pass

    def cross_validation(self, clf_model, trainset_features, trainset_target):
        """
        Метод реализующий процесс кросс-валидации
        """
        logger.debug(f'Начинаем кросс-валидацию. Количество разбиений: {СVSplitsNumber}')
        kfold = KFold(n_splits=СVSplitsNumber, shuffle=True)
        cv_scores = []
        ind = 0
        for train_index, test_index in kfold.split(trainset_features):
            train_x = trainset_features.iloc[train_index]
            val_x = trainset_features.iloc[test_index]
            train_y = trainset_target.iloc[train_index]
            val_y = trainset_target.iloc[test_index]
            clf_model.fit(train_x, train_y)
            predicted_test_y = clf_model.predict(val_x)
            logger.debug(f'Current accuracy score ({ind}): {accuracy_score(val_y, predicted_test_y)}')
            cv_scores.append(accuracy_score(val_y, predicted_test_y))
        logger.debug(f'cv_scores: {cv_scores}')
        cross_validation_mean_accuracy = np.mean(cv_scores)
        logger.info(f'Средняя accuracy на кросс-валидации: {cross_validation_mean_accuracy}')
        return cross_validation_mean_accuracy

    def fit_model(self, train_x, train_y):
        """
        Метод тренировки модели
        """
        logger.info('Обучаем итоговую версию модели')
        self.classifier_model.fit(train_x, train_y)
        logger.info('Итоговая версия модели обучена!')

    def check_quality(self, test_x, test_correct_y):
        """
        Метод оценки качества классификациии обученной модели
        """
        test_predicted_labels = self.classifier_model.predict(test_x)
        accuracy_test = accuracy_score(test_correct_y, test_predicted_labels)
        logger.info(f"Итоговое accuracy модели с оптимальными гиперпараметрами на тестовой выборке:\n{accuracy_test}")
        return accuracy_test

    @staticmethod
    def trainset_train_test_split(trainset_features, trainset_target, test_size_value: float = 0.2, print_logs: bool = False):
        """
        Метод разбиения трейнсета на обучающую и тестовую выборки

        test_size - отношение размера тестовой выборки (по умолчанию = 0.3)
        """
        if print_logs:
            logger.info(f'Разбиваем выборку на train/test. Размер теста - {test_size_value}')

        train_x, test_x, train_y, test_y = train_test_split(
            trainset_features,
            trainset_target,
            test_size=test_size_value
        )
        return train_x, test_x, train_y, test_y

    def predict_classes(self, products_for_classification):
        """
        Метод выполняющий классификацию для переданных ему данных
        """
        logger.info('Выполним итоговое предскакзание классов')
        predicted_classes = self.classifier_model.predict(products_for_classification)
        predicted_classes_df = pd.DataFrame(predicted_classes)
        return predicted_classes_df

