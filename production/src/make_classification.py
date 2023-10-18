import pandas as pd
from loguru import logger
from TrainsetHandler import TrainsetHandler

class Classificator:
    """
    Класс для выполнения полного цикла классификации номенклатур: 
        1. Первичная обработка данных
        2. Обработка факторов трейнсета
        3. Обучение модели 
        4. Предсказание результата
    
    Parameters
    ----------
    products_df: pandas.DataFrame - таблица с данными для предсказания.
        Должна содержать столбцы:
            - "ID класса (ТАРГЕТ)" - с id предсказываемых классов
            - "Историческое наименование"
            - Пары колонок с названями в формате: "ХК_{тип данных}_номер" и "Значение ХК_{тип данных}_номер"
            - Все остальные столбцы будут проигнорированны 
    """
    # Структура классов:
    # * Класс для общего workflow (ClassificatorWorkFlow)
    #   * Класс для хранения данных нашего решения (изначально сырые данные, потом трейнсет) (DataHandler)
    #      * Класс для сырых данных (данные и методы их обработки) (RawDataHandler)
    #      * Класс для трейнсета и его (обработки) (TrainsetHandler)
    #   * Интерфейс для модели (чтобы можно было добавлять другие алгоритмы, не только RandomForest) (ClassificatorModelInterface)
    #       * Класс для RandomForest (RandomForestModel)
    #   
    # Декораторы:
    # * Логгер
    # * Обработчик ошибок

    # todo Добавить:
    # * Общую структуру и запушить
    # * ЧТО ДЕЛАТЬ В СЛУЧАЕ ОШИБОК ПРИ ВЫПОЛНЕНИИ?
    # * Логгирование
    # * ProductsData - Класс для всего что касается хранения и данных нашего решения (сырые данные)
    # * DataHandler - Класс первичной обработки сырых данных (выделение типов колонок, заполнение пропусков, кодирование переменных)
    # * TrainsetHandler - Класс для хранения и обработки трейнсета (Хранение трейнсета, таргета(данные для которых будем строить итоговое предсказание), данных для предсказания, названий колонок и типов данных. 
    #                                             Удаление слабо коррелирующих с таргетом факторов, несбалансированность классов, PCA, разбиение на train_test)
    # * Classificator Класс модели классификатора (подборка гиперпараметров, обучение моделей, оценка качества, итоговое предсказание ) 
    # * Проверку на известный тип данных колонки

    def __init__(self, raw_data: pd.DataFrame = None):
        self.raw_data = raw_data

        self.trainset = None
        # todo Итоговое предсказание и метрики качества (возможно лучше вынести в класс)
        self.final_prediction = None
        self.final_metrics = None

    def classify_products() -> pd.DataFrame:
        """
        Основной метод запускающий все этапы генерации данных
        """
        logger.info("Начало работы алгоритма.")
        # Обработка сырых данных
        raw_data_handling()
        # Формирование трейнсета
        trainset_forming()
        # Обучение и выбор модели
        classificator_fit()
        # Итоговое предсказание
        final_prediction = classificator_predict()
        return final_prediction
    
    def raw_data_handling():
        a = 1
    
    def trainset_forming():
        a = 1

    def classificator_fit():
        a = 1

    def classificator_predict():
        a = 1


raw_data_df = pd.read_excel('data/tire_classificator_data.xlsx')
target_predict = ClassificatorWorkFlow(raw_data_df).classify_products()
