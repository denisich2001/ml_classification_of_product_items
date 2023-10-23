import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from production.src.handle_raw_product_table import RawProductsTableHandler
from production.src.utils.errors import NoPredictionsDataException
from production.config import TargetNameColumn

class TrainsetHandler:
    """
    Класс для обработки сырых данных, формирования и хранения трейнсета:
        * Хранение сформированного трейнсета, таргета
        * Обработка сырых данных
        * Хранение данные для которых будем строить итоговое предсказание
        * Отбор только информативных и некоррелирующих факторов
        * Устранение несбалансированности классов
        * PCA
    """
    def __init__(self, raw_product_table: pd.DataFrame) -> None:
        self.raw_product_table = raw_product_table.copy()

        # Переменные для итогового трейнсета после всех обработок
        self.trainset_features = None
        self.trainset_target = None

        # Данные для которых будет выполняться итоговое предсказание
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
        primary_trainset = self.separate_predictions_data_from_train(handled_product_table)
        informative_factors_trainset = self.informative_factors_selection(primary_trainset)
        balanced_trainset = self.trainset_target_balancing(informative_factors_trainset)
        final_trainset = self.pca_trainset_transformation(balanced_trainset)

        self.trainset_target = final_trainset[TargetNameColumn]
        self.trainset_features = final_trainset.drop(TargetNameColumn, axis=1)
        logger.info('Трейнсет успешно сформирован.')
        return self.trainset_features, self.trainset_target

    def separate_predictions_data_from_train(self, original_product_table: pd.DataFrame):
        """
        Метод отделения тренировочных данных от данных для предсказания (строки с пустыми значениями в колонке класса)
        """
        logger.info('Отделяем тренировочные данные от данных, для которых нужно выполнить предсказание класса.')
        product_table = original_product_table.copy()
        self.products_for_classification = product_table[product_table['ID класса (ТАРГЕТ)'].isna()]
        product_table_for_train = product_table[product_table['ID класса (ТАРГЕТ)'].notna()]
        logger.info(f'{self.products_for_classification.shape[0]} строк для выполнения классификации.')
        return product_table_for_train

    def informative_factors_selection(self, original_trainset: pd.DataFrame):
        # TODO ДОДЕЛАТЬ ОТБОР ФАКТОРОВ ПО КОРРЕЛЯЦИИ И ИНФОРМАТИВНОСТИ И ДИСПЕРСИИ
        return original_trainset

    def trainset_target_balancing(self, original_trainset: pd.DataFrame):
        logger.info('Устраним несбалансированность классов.')
        trainset = original_trainset.copy()
        logger.debug(f'2 самых частых и самых редких класса до балансировки:\n'
                     f'{trainset.value_counts()[:2]}\n{trainset.value_counts()[-2:]}')
        ros = RandomOverSampler()
        balanced_features, balanced_target = ros.fit_resample(
            trainset.drop('TargetNameColumn', axis=1),
            trainset[TargetNameColumn]
        )
        balanced_trainset = pd.concat([balanced_features, balanced_target], axis=1)
        logger.info('Устранили несбалансированность классов.')
        logger.debug(f'2 самых частых и самых редких класса после балансировки:\n'
                     f'{balanced_trainset.value_counts()[:2]}\n{balanced_trainset.value_counts()[-2:]}')
        return balanced_trainset

    def pca_trainset_transformation(self, original_trainset: pd.DataFrame):
        # TODO ДОДЕЛАТЬ PCA (подбор оптималного количества компонент, pca, результаты)
        pca = PCA(n_components=15)