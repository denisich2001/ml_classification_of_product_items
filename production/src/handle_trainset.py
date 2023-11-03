import numpy as np
import pandas as pd
from loguru import logger

from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer

from production.config import TargetColumnName
from production.config import PCAComponentsNumber
from production.src.utils.errors import EmptyValuesAfterEncoding
from production.src.utils.features_preprocessing import text_feature_preprocessing

russian_stopwords = stopwords.words("russian")


class DataHandler:
    """
    Класс подготовки данных для работы с моделью:
    * Кодирование факторов
    * Снижение размерности с помощью PCA
    * Устранение несбалансированности классов в тренировочной выборке
    """
    def __init__(
            self, handled_product_table, input_table_types_dict,
    ) -> None:
        self.input_table_types_dict = input_table_types_dict
        self.handled_product_table = handled_product_table
        # Переменные для итогового трейнсета после всех обработок
        self.final_trainset = None
        self.predictions_dataset = None

    def prepare_dataset(self):
        """
        Главный метод подготовки данных для тренировки и предсказания.
        """
        logger.info('Начинаем подготовку данных для обучения модели.')
        primary_dataset = self.handled_product_table.copy()
        dataset_target = primary_dataset[TargetColumnName]
        primary_dataset_features = primary_dataset.drop(TargetColumnName, axis=1)
        encoded_dataset_features = self.encode_features(primary_dataset_features)
        reduced_dataset_features = self.pca_transformation(encoded_dataset_features)
        reduced_dataset = pd.concat([reduced_dataset_features, dataset_target], axis=1)
        reduced_trainset, self.predictions_dataset = self.separate_predictions_data_from_train(reduced_dataset)
        self.predictions_dataset = self.predictions_dataset.drop(TargetColumnName, axis=1)
        balanced_trainset = DataHandler.trainset_target_balancing(reduced_trainset)
        self.final_trainset = balanced_trainset
        logger.info('Данные подготовлены.')
        return self.final_trainset, self.predictions_dataset

    @staticmethod
    def trainset_target_balancing(original_dataset: pd.DataFrame):
        """
        Метод устранения несбалансированности классов. Используем пересемплирование.
        """
        logger.info('Устраним несбалансированность классов.')
        dataset = original_dataset.copy()
        ros = RandomOverSampler()
        balanced_features, balanced_target = ros.fit_resample(
            dataset.drop(TargetColumnName, axis=1),
            dataset[TargetColumnName]
        )
        balanced_dataset = pd.concat([balanced_features, balanced_target], axis=1)
        logger.info('Устранили несбалансированность классов.')
        return balanced_dataset

    def pca_transformation(self, original_dataset: pd.DataFrame):
        """
        Метод снижения размерности трейнсета методом главных компонент.
        """
        dataset = original_dataset.copy()
        pca = PCA(n_components=PCAComponentsNumber)
        reduced_dataset = pd.DataFrame(pca.fit_transform(dataset))
        return reduced_dataset

    def encode_features(self, original_dataset: pd.DataFrame):
        """
        Метод кодирования строковых и категориальных факторов модели
        """
        logger.info('Начинаем кодирование факторов.')
        dataset = original_dataset.copy()
        dataset_encoded = pd.DataFrame(index=original_dataset.index)
        for feature in self.input_table_types_dict:
            if self.input_table_types_dict.get(feature) == 'Стр':
                handled_feature = self.handle_text_feature(dataset[feature])
            elif self.input_table_types_dict.get(feature) == 'Кат':
                handled_feature = self.handle_cat_feature(dataset[feature])
            else:
                handled_feature = pd.DataFrame(dataset[feature])
            handled_feature.columns = [str(col) + '_' + str(feature) for col in handled_feature.columns]
            if self.input_table_types_dict.get(feature) == 'Кат':
                dataset_encoded = dataset_encoded.join(handled_feature)
        # Т.к. мы пересобирали датасет заново во время кодирования => перепроверим на наличие пропущеных значений
        if dataset_encoded.isna().sum().sum() > 0:
            logger.error('Появились пустые значения после кодирования переменных')
            raise EmptyValuesAfterEncoding('Появились пустые значения после кодирования переменных!')
        logger.info(f'Получилось {dataset_encoded.columns.size} факторов после кодирования')
        return dataset_encoded

    def handle_cat_feature(self, cat_feature: pd.Series):
        """
        Метод кодирования категориального фактора - используем OneHotEncoding
        """
        cat_feature = cat_feature.astype(str).values.reshape(-1, 1)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(cat_feature)
        cat_feature_encoded = pd.DataFrame(encoder.transform(cat_feature))
        final_cat_feature_encoded = cat_feature_encoded.drop(
            [col for col in cat_feature_encoded.columns if cat_feature_encoded[col].sum() == 0],
            axis=1
        )
        return final_cat_feature_encoded

    def handle_text_feature(self, text_feature: pd.Series):
        """
        Метод обработки и кодирования строкового фактора:
        - Проводим препроцессинг
        - Используем TFidfVectorizer
        """
        processed_text_feature = text_feature_preprocessing(text_feature)
        vectorizer = TfidfVectorizer(
                max_features=100,
                analyzer='word',
                stop_words=russian_stopwords+["emptyvalue"]
        )
        vectorized_text_feature = pd.DataFrame(vectorizer.fit_transform(processed_text_feature).toarray())
        vectorized_text_feature.columns = pd.Series(vectorizer.get_feature_names_out())
        return vectorized_text_feature

    @staticmethod
    def separate_predictions_data_from_train(original_product_table: pd.DataFrame):
        """
        Метод отделения тренировочных данных от данных для предсказания (строки с пустыми значениями в колонке класса)
        """
        logger.info('Отделяем тренировочные данные от данных, для которых нужно выполнить предсказание класса.')
        product_table = original_product_table.copy()
        products_for_classification = product_table[
            product_table[TargetColumnName].isna()]
        product_table_for_train = product_table[product_table[TargetColumnName].notna()]
        logger.info(f'{products_for_classification.shape[0]} строк для выполнения классификации.')
        logger.debug(f'Пустых значений в данных для классификации:\n{products_for_classification.isna().sum()}')
        return product_table_for_train, products_for_classification