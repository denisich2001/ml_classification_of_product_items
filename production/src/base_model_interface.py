import numpy as np
import optuna
from abc import ABC
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from production.src.handle_trainset import DataHandler
from production.config import СVSplitsNumber
from production.config import TargetColumnName


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
            input_table_types_dict: dict
    ) -> None:
        self.main_data_handler = DataHandler(input_table_types_dict)
        self.trainset_target = product_table_for_train[TargetColumnName]
        self.trainset_features = product_table_for_train.drop(TargetColumnName, axis=1)
        self.input_table_types_dict = input_table_types_dict

        self.classifier_model = classifier_model
        self.vectorizer = None
        self.model_best_params = None
    # TODO ОЦЕНКА КАЧЕСТВА
    # TODO ПОДБОР ГИПЕРПАРАМЕТРОВ

    def prepare_model(self):
        """
        Основной метод, реализующий весь процесс взаимодействия с моделью.
        * Разбиение на train/test
        * Обучение модели - тьюнинг гиперпараметров с кросс-валидацией
        * Обучение модели с лучшими параметрами
        * Оценка качества классификациии
        """
        train_x, test_x, train_y, test_y = self.trainset_train_test_split()
        self.params_tuner(train_x, train_y)
        self.fit_model(train_x, train_y)
        self.check_quality(test_x, test_y)

    def params_tuner(self, train_x, train_y, n_trials: int = 10):
        """
        Метод обучения модели:
        * Тьюнинг гиперпараметров с кросс-валидацией
        * Определение лучших гиперпараметров и обучение модели на них
        """
        logger.info('Подбираем оптимальные гиперпараметры для модели')
        study = optuna.create_study(direction="maximize")
        logger.debug(f'Размеры тренировочной выборки для кросс-валидации: {train_x.shape}')
        objective_func = lambda trial: self.objective(trial, train_x, train_y)
        study.optimize(objective_func, n_trials=n_trials)
        # Сохраняем лучшие гиперпараметры
        self.model_best_params = study.best_params
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
        cv_scores = np.array
        for train_index, test_index in kfold.split(trainset_features):
            train_data = pd.concat([trainset_features.iloc[train_index], trainset_target.iloc[train_index]], axis=1)
            test_data = pd.concat([trainset_features.iloc[test_index], trainset_target.iloc[test_index]], axis=1)
            temp_data_handler = DataHandler(self.input_table_types_dict)
            train_x, train_y = temp_data_handler.prepare_traindata(train_data)
            test_x, test_y = temp_data_handler.prepare_prediction_or_test_data(test_data)
            logger.debug(f'Размеры выборок: {train_x.shape}, {train_y.size}, {test_x.shape}, {test_y.size}')
            clf_model.fit(train_x, train_y)
            predicted_test_y = clf_model.predict(test_x)
            np.append(cv_scores, accuracy_score(test_y, predicted_test_y))
        cross_validation_mean_accuracy = cv_scores.mean()
        logger.info(f'Средняя accuracy на кросс-валидации: {cross_validation_mean_accuracy}')
        return cross_validation_mean_accuracy

    def fit_model(self, original_train_x, original_train_y):
        """

        """
        original_train_data = pd.concat([original_train_x, original_train_y], axis=1)
        temp_data_handler = DataHandler(self.input_table_types_dict)
        train_x, train_y = temp_data_handler.prepare_prediction_or_test_data(original_train_data, print_logs=True)
        self.classifier_model.fit(train_x, train_y)

    def predict_classes(self, original_products_for_classification):
        """
        Метод выполняющий классификацию для переданных ему данных
        """
        logger.info('Выполним классификацию')
        products_for_classification, empty_classes = self.main_data_handler.prepare_prediction_or_test_data(
            original_products_for_classification,
            print_logs=True
        )
        predicted_classes = self.classifier_model.predict(products_for_classification)
        predicted_classes_df = pd.DataFrame(predicted_classes)
        return predicted_classes_df

    def check_quality(self, test_x, test_correct_y):
        """
        Метод оценки качества классификациии обученной модели
        """
        # TODO Добавить TFidVectorizer сюда

        # TODO Добавить PCA сюда
        #train_predicted_labels = self.classifier_model.predict(train_x)
        test_predicted_labels = self.classifier_model.predict(test_x)

        #accuracy_train = accuracy_score(train_correct_y, train_predicted_labels)
        accuracy_test = accuracy_score(test_correct_y, test_predicted_labels)
        #report = classification_report(test_correct_y, test_predicted_labels)

        #logger.debug(f"Accuracy на тренировочной выборке:\n{accuracy_train}")
        logger.debug(f"Accuracy на тестовой выборке:\n{accuracy_test}")
        #print("Classification Report:\n", report)

    def trainset_train_test_split(self, test_size_value: float = 0.3, print_logs: bool = False):
        """
        Метод разбиения трейнсета на обучающую и тестовую выборки

        test_size - отношение размера тестовой выборки (по умолчанию = 0.3)
        """
        if print_logs:
            logger.info(f'Разбиваем выборку на train/test. Размер теста - {test_size_value}')

        print(f'Размеры начального датасета: {self.trainset_features.shape}')
        print(f'Размеры начального датасета: {self.trainset_target}')

        train_x, test_x, train_y, test_y = train_test_split(
            self.trainset_features,
            self.trainset_target,
            test_size=test_size_value
        )
        return train_x, test_x, train_y, test_y

