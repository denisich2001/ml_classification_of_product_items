import pandas as pd
from loguru import logger
from handle_trainset import TrainsetHandler
from handle_raw_products_frame import RawProductsFrameHandler
from utils.errors import NoProductsDataException

class Classificator:
    """
    Класс для выполнения полного цикла классификации номенклатур: 
        1. Первичная обработка данных
        2. Обработка факторов трейнсета
        3. Обучение модели 
        4. Предсказание результата
    
    Parameters
    ----------
    raw_products_frame: pandas.DataFrame - таблица с данными для предсказания.
        Должна содержать столбцы:
            - "ID класса (ТАРГЕТ)" - с id предсказываемых классов
            - "Историческое наименование"
            - Пары колонок с названями в формате: "ХК_{тип данных}_номер" и "Значение ХК_{тип данных}_номер"
            - Все остальные столбцы будут проигнорированны
    n_workers - количество процессоров доступных для выполнения кода (по умолчанию = 1)
    """
    # todo Добавить:
    # * Проверку на известный тип данных колонки
    # * Все числовые факторы перевожу в категориальные - протестировать
    # * Декоратор для обработки ошибок
    # * FastApi
    # * Добавить логгирование

    def __init__(self, raw_products_frame: pd.DataFrame = None, n_workers: int = 1):
        # todo ЛОГГИРОВАНИЕ

        self.raw_products_frame = raw_products_frame
        self.trainset = None

        # todo Итоговое предсказание и метрики качества (возможно лучше вынести в класс)
        self.final_prediction = None
        self.final_metrics = None

        self.n_workers = n_workers


    def classify_products(self) -> pd.DataFrame:
        """
        Основной метод, запускающий все этапы генерации данных
        """
        logger.info("Начало работы алгоритма.")
        self.input_parameters_check()
        self.handle_products_data()
        self.classificator_fit()
        final_prediction = self.classificator_predict()
        return final_prediction
    
    def input_parameters_check(self) -> bool:
        """
        Проверка на наличие ошибок в исходных данных:
            * Проверка на наличие датафрейма с данными для обучения
            * Проверка количества доступных процессоров
        """
        if self.raw_products_frame is None:
            raise NoProductsDataException()
        # todo Проверка количества доступных процессоров
        return True

    def handle_products_data(self):
        """
        Проверка на возникновение ошибок при формировании трейнсета
        """
        try:
            self.trainset = TrainsetHandler(self.raw_products_frame).form_trainset()
        except:
            # todo добавить обработку исключений
            logger.error('Какая-то ошибка при формаировании трейнсета!')
        finally:
            # todo изменить
            exit()

    def classificator_fit(self):
        a = 1

    def classificator_predict(self):
        a = 1