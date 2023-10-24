import pandas as pd
import numpy as np
from loguru import logger
from production.src.handle_trainset import DataHandler
from production.src.handle_raw_product_table import RawProductsTableHandler
from production.src.utils.errors import NoProductsDataException
from production.src.utils.errors import NoTargetColumnException
from production.src.utils.errors import NoPredictionsDataException
from production.config import TargetNameColumn


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

    # todo Добавить:
    # * Проверить работу exceptions
    # * Проверку на известный тип данных колонки
    # * Декоратор для обработки ошибок
    # * FastApi
    # * Добавить логгирование

    def __init__(self, raw_product_table: pd.DataFrame = None, n_workers: int = 1):
        self.raw_product_table = raw_product_table
        self.data_handler = None
        self.data_classifier = None
        # todo Итоговое предсказание и метрики качества (возможно лучше вынести в класс)
        self.final_prediction = None
        self.final_metrics = None

        self.n_workers = n_workers

    def classify_products(self):
        """
        Основной метод, запускающий все этапы генерации данных
        """
        logger.info("Начало работы алгоритма.")
        self.input_parameters_check()
        self.data_handler = DataHandler(self.raw_product_table)
        trainset_features, trainset_target, products_for_classification = self.data_handler.form_trainset()
        return trainset_features, trainset_target
        # self.classificator_fit()
        # final_prediction = self.classificator_predict()
        # return final_prediction

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

        if TargetNameColumn not in self.raw_product_table.columns:
            logger.error('Отсутствует колонка с идентификатором класса для предсказания!\n'
                         f'Указанное название столбца с классом для предсказания: {TargetNameColumn}')
            raise NoTargetColumnException('Отсутствует колонка с идентификатором класса для предсказания!')

        data_for_predictions = self.raw_product_table[self.raw_product_table[TargetNameColumn].isna()]
        if (data_for_predictions is None) or (data_for_predictions.size == 0):
            logger.error('Отсутвуют данные для выполнения предсказания класса!')
            raise NoPredictionsDataException('Отсутвуют данные для выполнения предсказания класса!')
        return True

    def classificator_fit(self):
        self.data_classifier = Ran

    def classificator_predict(self):
        a = 1
