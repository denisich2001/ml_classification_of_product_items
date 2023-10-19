import re
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from utils.errors import UnknownColumnTypeException
from utils.errors import IncorrectColumnNameException
from utils.errors import NumberFeatureException
from utils.errors import EmptyValuesAfterEncoding
from utils.feature_processing import text_feature_preprocessing
from config import OneHotEncodingLimit

class RawProductsFrameHandler:
    '''
    Класс первичной обработки сырых данных:
        * Удаление лишних колонок
        * Выделение типов колонок
        * Заполнение пропусков
        * Кодирование переменных
    
    Parameters
    ----------
    raw_products_frame - сырой (необработанный) датасет переданный на вход
    '''
    def __init__(self, raw_products_frame):
        self.raw_products_frame = raw_products_frame.copy()
        # Итоговый датасет, после всех этапов обработки
        self.final_products_frame = None

        self.features_type_dict = None

    def handle_raw_products_frame(self):
        """
        Основной метод, вызывающий все методы обработки сырых данных
        """
        # На время обработки отделим столбец с меткой класса
        products_target_column = self.raw_products_frame['ID класса (ТАРГЕТ)']
        products_frame = self.raw_products_frame.copy().drop('ID класса (ТАРГЕТ)', axis = 1)

        products_frame_without_extra_columns = self.delete_extra_columns(products_frame)
        products_frame_with_formatted_features = self.format_feature_types(products_frame_without_extra_columns)
        products_frame_without_na = self.fill_na_in_features(products_frame_with_formatted_features)
        self.final_products_frame = self.encode_features(products_frame_without_na)

        # Вернем столбец с меткой класса
        self.final_products_frame['ID класса (ТАРГЕТ)'] = products_target_column
        return self.final_products_frame
    
    def delete_extra_columns(self, original_products_frame):
        '''
        Метод удаления лишних столбцов. 
        Итоговый датафрейм содержит столбцы:
            * Историческое наименование
            * Пары колонок с названями в формате: "ХК_{тип данных}_номер" и "Значение ХК_{тип данных}_номер"
        '''
        products_frame = original_products_frame.copy()

        products_frame_columns = []
        for column in products_frame.columns:
            if (column == 'Историческое наименование') or (re.fullmatch(r'ХК_.*', column)!=None) or (re.fullmatch(r'Значение.*', column)!=None):
                products_frame_columns.append(column)

        products_frame_without_extra_columns = products_frame[products_frame_columns]
        return products_frame_without_extra_columns
    
    def format_feature_types(self, original_products_frame):
        """
        Метод задающий типы данных колонкам
        """
        products_frame = original_products_frame.copy()
        self.feature_types_dict = self.get_column_types_dict(products_frame.columns)

        for feature in feature_types_dict.keys():
            if feature_types_dict.get(feature) == 'Стр':
                #todo ДОДЕЛАТЬ пока ничего не делаем, чтобы не потерять пропущенные значения при преобразовании в строковый формат
                products_frame[feature] = products_frame.copy()[feature].astype(object)
            elif feature_types_dict.get(feature) == 'Булево':
                products_frame[feature] = products_frame.copy()[feature].astype(bool)
            elif feature_types_dict.get(feature) == 'Числ':
                # Т.к. в номенклатурах нет как таковых числовых факторов => переводим их в категориальные
                feature_types_dict[feature] = 'Кат'
                products_frame[feature] = products_frame.copy().astype(object)
            elif feature_types_dict.get(feature) == 'Кат':
                products_frame[feature] = products_frame.copy().astype(object)
        return products_frame

    def get_column_types_dict(columns: list):
        '''
        Функция, выделяющая тип данных из названий колонок.
        Возвращает словарь с парами: название колонки - ее тип данных  
        '''
        feature_types_dict = {}
        for column in columns:
            type_pattern = r'ХК_([^_]+)_.*'
            if column[0:2] == 'ХК':
                feature_types_dict[column] = 'Кат'
            elif column[0:8]=='Значение':
                column_type = re.findall(type_pattern, column)[0]
                if column_type not in ['Кат','Стр','Числ','Булево']:
                    logger.error('Неизвестный тип данных в названии столбца!')
                    raise UnknownColumnTypeException('Неизвестный тип данных в названии столбца!')
                feature_types_dict[column] = column_type
            else:
                # Если неизвестный тип данных в названии колонки - выкидываем исключение
                logger.error('Неправильный формат названия характеристики!')
                raise IncorrectColumnNameException('Неправильный формат названия характеристики!')
            
        return feature_types_dict

    def fill_na_in_features(self, original_products_frame):
        """
        В зависимости от типа данных колонки заполняем пропуски по-разному:
        * Стр - ''
        * Числ - перевели все численные столбцы в категориальные 
        * Булево - булевые столбцы почти всегда отвечают на вопрос: "Есть что-то? - Да/Нет". Если нет ответа => Нет
        * Кат - 'Emptyclass'
        """
        products_frame = original_products_frame.copy()
        for column in products_frame.columns:
            if self.feature_types_dict.get(column) == 'Кат':
                products_frame.loc[products_frame[column].isna(), column] = f'EmptyCat'
            elif self.feature_types_dict.get(column) == 'Стр':
                # todo Перепроверить норм ли задавать пустое значение строке как пустую строку
                products_frame.loc[products_frame[column].isna(), column] = ''
            elif self.feature_types_dict.get(column) == 'Булево':
                products_frame.loc[products_frame[column].isna(), column] = 0
            elif self.feature_types_dict.get(column) == 'Числ':
                raise NumberFeatureException('Присутствует числовой фактор, который должен быть переведен в категориальный!')
        return products_frame
    
    def encode_features(self, original_products_frame):
        products_frame = original_products_frame.copy()
        # todo ДОДЕЛАТЬ
        products_frame_encoded = pd.DataFrame()

        for feature in self.feature_types_dict:
            if self.feature_types_dict.get(feature) == 'Стр':
                handled_feature = self.handle_text_feature(products_frame[feature])
                #handled_feature = handled_feature.fillna('')
                # todo ПРОВЕРИТЬ НА ПРОПУСКИ
            elif self.feature_types_dict.get(feature) == 'Кат':
                handled_feature = self.handle_cat_feature(products_frame[feature])
            else:
                handled_feature = pd.DataFrame(products_frame[feature])
            handled_feature.columns = [str(col)+'_'+str(feature) for col in handled_feature.columns]  
            products_frame_encoded = pd.concat([products_frame_encoded,handled_feature],axis=1)

        # Т.к. мы пересобирали датасет заново во время кодирования => перепроверим на наличие пропущеных значений
        if products_frame_encoded.isna().sum().sum()>0:
            logger.error('Появились пустые значения после кодирования переменных')
            raise EmptyValuesAfterEncoding('Появились пустые значения после кодирования переменных!')
        return products_frame_encoded

    def handle_cat_feature(cat_feature: pd.Series):
        cat_feature = cat_feature.astype(str)
        unique_values_count = cat_feature.drop_duplicates().size
        if unique_values_count <= OneHotEncodingLimit:
            #OneHotEncoding
            cat_feature_encoded = pd.get_dummies(cat_feature)
        else:
            #LabelEncoding - чтобы сильно не увеличивать количество факторов
            le = LabelEncoder()
            cat_feature_encoded = pd.DataFrame(le.fit_transform(cat_feature))
        return cat_feature_encoded

    def handle_text_feature(self, text_feature: pd.Series):
        '''
        Функция обработки строкового фактора:
        - Проводим препроцессинг
        - Формируем "Мешок строк" (bag of words)
        '''
        processed_text_feature = text_feature_preprocessing(text_feature)
        vectorizer = CountVectorizer()
        vectorizer.fit(processed_text_feature)
        vectorized_text_feature = pd.DataFrame(vectorizer.transform(processed_text_feature).toarray())

        # Удалим неинформативные столбцы
        informative_word_columns = vectorized_text_feature.sum()[
                (vectorized_text_feature.sum()>=vectorized_text_feature.shape[1]*0.01) &
                (vectorized_text_feature.sum()!=vectorized_text_feature.shape[1])
            ].index 
        handled_text_feature = vectorized_text_feature[informative_word_columns]
        #handled_text_feature = handled_text_feature.fillna('EmptyStr')
        handled_text_feature.columns = pd.Series(vectorizer.get_feature_names_out())[informative_word_columns]
        return handled_text_feature