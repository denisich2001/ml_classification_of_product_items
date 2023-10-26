import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from production.src.handle_raw_product_table import RawProductsTableHandler
from production.src.utils.errors import NoPredictionsDataException
from production.config import TargetNameColumn


class DataHandler:
    """
    Класс для обработки сырых данных, формирования и хранения трейнсета:
        * Хранение сформированного трейнсета, таргета
        * Обработка сырых данных
        * Хранение данные для которых будем строить итоговое предсказание класса
        * Устранение несбалансированности классов
        * PCA
    """
    def __init__(self, raw_product_table: pd.DataFrame) -> None:
        self.raw_product_table = raw_product_table.copy()

        # Переменные для итогового трейнсета после всех обработок
        self.trainset_features = None
        self.trainset_target = None

        # Данные для которых будет выполняться итоговое предсказание класса
        self.products_for_classification = None
         
    def form_trainset(self):
        """
        Основной метод формирования трейнсета
        Returns
        -------
        trainset_features и trainset_target - факторы и таргет сформированного трейнсета
        """
        logger.info('Начинаем формирование трейнсета.')
        handled_product_table = RawProductsTableHandler(self.raw_product_table).handle_raw_product_table()
        primary_trainset, products_for_classification = DataHandler.separate_predictions_data_from_train(
            handled_product_table
        )
        balanced_trainset = DataHandler.trainset_target_balancing(primary_trainset)
        self.trainset_target = balanced_trainset[TargetNameColumn]
        balanced_trainset = balanced_trainset.drop(TargetNameColumn, axis=1)
        final_trainset, final_products_for_classification = DataHandler.pca_transformation(
            balanced_trainset,
            products_for_classification
        )
        self.trainset_features = final_trainset
        self.products_for_classification = final_products_for_classification
        logger.info('Трейнсет успешно сформирован.')
        return self.trainset_features, self.trainset_target, self.products_for_classification

    @staticmethod
    def separate_predictions_data_from_train(original_product_table: pd.DataFrame):
        """
        Метод отделения тренировочных данных от данных для предсказания (строки с пустыми значениями в колонке класса)
        Parameters
        ----------
        original_product_table - основной датасет (пока трейнсет и данные для предсказания - вместе)
        """
        logger.info('Отделяем тренировочные данные от данных, для которых нужно выполнить предсказание класса.')
        product_table = original_product_table.copy()
        products_for_classification = product_table[product_table[TargetNameColumn].isna()].drop(TargetNameColumn, axis=1)
        product_table_for_train = product_table[product_table['ID класса (ТАРГЕТ)'].notna()]
        logger.info(f'{products_for_classification.shape[0]} строк для выполнения классификации.')
        logger.debug(f'Пустых значений в данных для классификации:\n{products_for_classification.isna().sum()}')
        return product_table_for_train, products_for_classification

    @staticmethod
    def trainset_target_balancing(original_trainset: pd.DataFrame):
        """
        Метод устранения несбалансированности классов. Используем пересемплирование.
        Parameters
        ----------
        original_trainset - основной трейнсет
        """
        logger.info('Устраним несбалансированность классов.')
        trainset = original_trainset.copy()
        logger.debug(f'2 самых частых и самых редких класса до балансировки:\n'
                     f'{trainset.value_counts()[:2]}\n{trainset.value_counts()[-2:]}')
        ros = RandomOverSampler()
        balanced_features, balanced_target = ros.fit_resample(
            trainset.drop(TargetNameColumn, axis=1),
            trainset[TargetNameColumn]
        )
        balanced_trainset = pd.concat([balanced_features, balanced_target], axis=1)
        logger.info('Устранили несбалансированность классов.')
        return balanced_trainset

    @staticmethod
    def pca_transformation(original_trainset: pd.DataFrame, original_products_for_classification: pd.DataFrame):
        """
        Метод снижения размерности трейнсета методом главных компонент.
        Parameters
        ----------
        original_trainset - основной трейнсет
        original_products_for_classification - данные для предсказания
        """
        trainset = original_trainset.copy()
        products_for_classification = original_products_for_classification.copy()
        logger.info('Снизим размерность используя метод главных компонент.')
        logger.debug(f'Размерность трейнсета до: {trainset.shape}')
        pca = PCA(n_components=20)
        reduced_trainset = pd.DataFrame(pca.fit_transform(trainset))
        reduced_products_for_classification = pd.DataFrame(pca.transform(products_for_classification))
        logger.debug(f'Размерность трейнсета после: {reduced_trainset.shape}')
        logger.debug(f'PCA explained variance ratio:\n {pca.explained_variance_ratio_}')
        logger.debug(f'Размерность данных для классификации после снижения: {reduced_products_for_classification.shape}')
        return reduced_trainset, reduced_products_for_classification

