import pandas as pd
from handle_raw_products_frame import RawProductsFrameHandler
from sklearn.model_selection import train_test_split
from utils.errors import NoPredictionsDataException
from loguru import logger

class TrainsetHandler:
    '''
    Класс для обработки сырых данных, формирования и хранения трейнсета:
        * Хранение сформированного трейнсета, таргета 
        * Хранение данные для которых будем строить итоговое предсказание
        * Обработка сырых данных
        * Удаление слабо коррелирующих с таргетом факторов
        * Несбалансированность классов 
        * PCA 
        * Разбиение на train_test
    '''
    def __init__(self, raw_products_frame: pd.DataFrame) -> None:
        self.raw_products_frame = raw_products_frame.copy()

        # Переменные для итогового трейнсета после всех обработок
        self.final_features = None
        self.final_target = None

        # Данные для которых будет выполняться итоговое предсказание
        self.products_for_classification = None
         
    def form_trainset(self):
        """
        Основной метод формирования трейнсета
        """
        logger.info('Начинаем формирование трейнсета.')
        handled_products_frame = RawProductsFrameHandler(self.raw_products_frame).handle_raw_products_frame()
        products_frame = self.separate_predictions_data_from_train(handled_products_frame)
        # todo ДОПИЛИТЬ
        return self.products_train_test_split()

    def separate_predictions_data_from_train(self, original_products_frame: pd.DataFrame):
        """
        Метод отделения тренировочных данных от данных для предсказания (строки с пустыми значениями в колонке класса)
        """
        products_frame = original_products_frame.copy()
        self.products_for_classification = products_frame[products_frame['ID класса (ТАРГЕТ)'].isna()]
        products_frame_for_train = products_frame[products_frame['ID класса (ТАРГЕТ)'].isna()]
        
        # Проверяем, присутствуют ли данные для выполнение предсказания классов
        if (self.products_for_classification is None) or (self.products_for_classification.shape[0]==0):
            logger.error('Отсутвуют данные для выполнения классификации!')
            raise NoPredictionsDataException('Отсутвуют данные для выполнения классификации!')
        
        return products_frame_for_train

    def products_train_test_split(self, test_size: float = 0.3):
        """
        Метод разбиения трейнсета на обучающую и тестовую выборки

        test_size - отношение размера тестовой выборки (по умолчанию = 0.3)
        """
        features_train, features_test, target_train, target_test = train_test_split(
            self.final_trainset_features, 
            self.final_trainset_target, 
            test_size  
        ) 
        return features_train, features_test, target_train, target_test