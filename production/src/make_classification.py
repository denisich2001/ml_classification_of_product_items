import pandas as pd
import numpy as np
from loguru import logger

from production.config import TargetColumnName
from production.src.handle_trainset import DataHandler
from production.src.random_forest_model import RandomForestModel
from production.src.utils.errors import NoProductsDataException
from production.src.utils.errors import NoTargetColumnException
from production.src.utils.errors import NoPredictionsDataException
from production.src.handle_raw_product_table import RawProductsTableHandler


class Classifier:
    """
    Класс для выполнения полного цикла классификации номенклатур: 
        1. Первичная обработка данных
        2. Обработка факторов трейнсета
        3. Обучение модели 
        4. Предсказание результата
    
    Parameters
    ----------
    raw_product_table: pandas.DataFrame - таблица с данными для предсказания.
        Должна содержать столбцы:
            - С id предсказываемых классов. Название должно быть указано в config в переменной TargetColumnName (по умолчанию - 'Id Class')
            - "Историческое наименование"
            - Пары колонок с названями в формате: "ХК_{тип данных}_номер" и "Значение ХК_{тип данных}_номер"
            - Все остальные столбцы будут проигнорированны
    n_workers - количество процессоров доступных для выполнения кода (по умолчанию = 1)
    """
    def __init__(self, raw_product_table: pd.DataFrame = None, n_workers: int = 1):
        self.raw_product_table = raw_product_table
        self.data_handler = None
        self.data_classifier = None

        # todo Итоговое предсказание и метрики качества (возможно лучше вынести в класс)
        self.final_prediction = None
        self.final_accuracy = None

        self.n_workers = n_workers

    def classify_products(self):
        """
        Основной метод, запускающий все этапы генерации данных:
        * Обработка сырых данных
        * Отделение тренировочных данных от данных для предсказания
        * Подбор и обучение модели
        * Выполнение предсказания
        """
        logger.info("Начало работы алгоритма.")
        self.input_parameters_check()
        handled_product_table, input_table_types_dict = \
            RawProductsTableHandler(self.raw_product_table).handle_raw_product_table()
        data_handler = DataHandler(handled_product_table, input_table_types_dict)
        trainset, dataset_for_classification = data_handler.prepare_dataset()
        self.data_classifier = RandomForestModel(trainset)
        self.final_accuracy = self.data_classifier.prepare_model()
        self.final_prediction = self.data_classifier.predict_classes(dataset_for_classification)
        return {
            'dataframe': self.final_prediction,
            'accuracy': self.final_accuracy
        }

    def input_parameters_check(self) -> bool:
        """
        Проверка на наличие ошибок в исходных данных:
            * Проверка на наличие датафрейма с данными для обучения
            * Проверка на наличия колонки таргета (назвение указано в config в переменной TargetNameColumn)
            * Проверка на наличие данных для которых нужно строить предсказание
            * ИСПРАВИТЬ ОБРАБОТКУ ФАКТОРОВ: ХК_ И ЗНАЧЕНИЕ ОБЪДИНИТЬ, Т.К. НЕ УЧИТЫВАЕТСЯ ПОРЯДОК В МОДЕЛИ
        """
        if self.raw_product_table is None:
            logger.error('Отсутствуют данные обучения модели классификации!')
            raise NoProductsDataException('Отсутствуют данные обучения модели классификации!')

        if TargetColumnName not in self.raw_product_table.columns:
            logger.error('Отсутствует колонка с идентификатором класса для предсказания!\n'
                         f'Указанное название столбца с классом для предсказания: {TargetColumnName}')
            raise NoTargetColumnException('Отсутствует колонка с идентификатором класса для предсказания!')

        data_for_predictions = self.raw_product_table[self.raw_product_table[TargetColumnName].isna()]
        if (data_for_predictions is None) or (data_for_predictions.size == 0):
            logger.error('Отсутвуют данные для выполнения предсказания класса!')
            raise NoPredictionsDataException('Отсутвуют данные для выполнения предсказания класса!')
        return True
