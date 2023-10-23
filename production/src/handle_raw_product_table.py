import re
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from utils.errors import UnknownColumnTypeException
from utils.errors import IncorrectColumnNameException
from utils.errors import NumberFeatureException
from utils.errors import EmptyValuesAfterEncoding
from utils.features_preprocessing import text_feature_preprocessing
from production.config import OneHotEncodingLimit
from production.config import TargetNameColumn

class RawProductsTableHandler:
    '''
    Класс первичной обработки сырых данных:
        * Удаление лишних колонок
        * Выделение типов колонок
        * Заполнение пропусков
        * Кодирование переменных
    
    Parameters
    ----------
    raw_product_table - сырой (необработанный) датасет переданный на вход
    '''
    def __init__(self, raw_product_table: pd.DataFrame):
        self.raw_product_table = raw_product_table.copy()
        # Словарь с типами данных колонок первичного
        self.input_table_types_dict = None
        # Итоговый датасет, после всех этапов обработки
        self.final_product_table = None

    def handle_raw_product_table(self):
        """
        Основной метод, вызывающий все методы первичной обработки необработанного датасета
        """
        logger.info('Начинаем первичную обработку таблицы с продуктами.')
        # На время обработки отделим столбец с меткой класса
        products_target_column = self.raw_product_table[TargetNameColumn]
        product_table = self.raw_product_table.copy().drop('TargetNameColumn', axis=1)

        product_table_without_extra_columns = self.delete_extra_columns(product_table)
        product_table_with_formatted_features = self.format_feature_types(product_table_without_extra_columns)
        product_table_without_na = self.fill_na_in_features(product_table_with_formatted_features)
        self.final_product_table = self.encode_features(product_table_without_na)

        # Вернем столбец с меткой класса
        self.final_product_table[TargetNameColumn] = products_target_column
        return self.final_product_table
    
    def delete_extra_columns(self, original_product_table: pd.DataFrame):
        '''
        Метод удаления лишних столбцов. 
        Итоговый датафрейм содержит столбцы:
            * Историческое наименование
            * Пары колонок с названями в формате: "ХК_{тип данных}_номер" и "Значение ХК_{тип данных}_номер"
        '''
        logger.info('Удаляем лишние столбцы.')
        product_table = original_product_table.copy()

        product_table_columns = []
        for column in product_table.columns:
            if (column == 'Историческое наименование') or (re.fullmatch(r'ХК_.*', column)!=None) or (re.fullmatch(r'Значение.*', column)!=None):
                product_table_columns.append(column)

        product_table_without_extra_columns = product_table[product_table_columns]
        logger.debug(f'Список столбцов после удаления: \n{list(product_table_without_extra_columns.columns)}')
        return product_table_without_extra_columns
    
    def format_feature_types(self, original_product_table: pd.DataFrame):
        """
        Метод задающий типы данных колонкам
        """
        logger.info('Задаем типы данных колонкам.')
        product_table = original_product_table.copy()
        self.input_table_types_dict = self.get_column_types_dict(product_table.columns)
        for feature in self.input_table_types_dict.keys():
            if self.input_table_types_dict.get(feature) == 'Стр':
                #todo ДОДЕЛАТЬ пока ничего не делаем, чтобы не потерять пропущенные значения при преобразовании в строковый формат
                product_table[feature] = product_table.copy()[feature].astype(object)
            elif self.input_table_types_dict.get(feature) == 'Булево':
                product_table[feature] = product_table.copy()[feature].astype(bool)
            elif self.input_table_types_dict.get(feature) == 'Числ':
                # Т.к. в номенклатурах нет как таковых числовых факторов => переводим их в категориальные
                self.input_table_types_dict[feature] = 'Кат'
                product_table[feature] = product_table.copy()[feature].astype(object)
            elif self.input_table_types_dict.get(feature) == 'Кат':
                product_table[feature] = product_table.copy()[feature].astype(object)
        logger.debug(f'Типы данных после редактирования: \n{product_table.dtypes}')
        return product_table

    def get_column_types_dict(self, columns: list):
        '''
        Функция, выделяющая тип данных из названий колонок.
        Возвращает словарь с парами: название колонки - ее тип данных
        '''
        logger.info('Выделяем тип данных из названий колонок.')
        input_table_types_dict = {}
        for column in columns:
            type_pattern = r'ХК_([^_]+)_.*'
            if column[0:2] == 'ХК':
                input_table_types_dict[column] = 'Кат'
            elif column[0:8]=='Значение':
                column_type = re.findall(type_pattern, column)[0]
                if column_type not in ['Кат','Стр','Числ','Булево']:
                    logger.error(f'Неизвестный тип данных в названии столбца: {column_type}!')
                    raise UnknownColumnTypeException(f'Неизвестный тип данных в названии столбца: {column_type}!')
                input_table_types_dict[column] = column_type
            elif column == 'Историческое наименование':
                input_table_types_dict[column] = 'Стр'
            else:
                # Если неизвестный тип данных в названии колонки - выкидываем исключение
                logger.error(f'Неправильный формат названия характеристики: {column}!')
                raise IncorrectColumnNameException(f'Неправильный формат названия характеристики: {column}!')
        logger.debug(f'Выделенные типы данных колонок: \n{input_table_types_dict}')
        return input_table_types_dict

    def fill_na_in_features(self, original_product_table: pd.DataFrame):
        """
        Метод заполнения пропусков в колонках.
        Для объектов с типом данных:
        * Стр - ''
        * Числ - перевели все численные столбцы в категориальные
        * Булево - булевые столбцы почти всегда отвечают на вопрос: "Есть что-то? - Да/Нет". Если нет ответа => Нет
        * Кат - 'Emptyclass'
        """
        logger.info('Заполняем пропуски')
        product_table = original_product_table.copy()
        logger.debug(f'Количество пустых значений в столбцах до заполнения пропусков: \n{product_table.isna().sum()}')
        for column in product_table.columns:
            if self.input_table_types_dict.get(column) == 'Кат':
                product_table.loc[product_table[column].isna(), column] = f'EmptyCat'
            elif self.input_table_types_dict.get(column) == 'Стр':
                # todo Перепроверить норм ли задавать пустое значение строке как пустую строку
                product_table.loc[product_table[column].isna(), column] = ''
            elif self.input_table_types_dict.get(column) == 'Булево':
                product_table.loc[product_table[column].isna(), column] = 0
            elif self.input_table_types_dict.get(column) == 'Числ':
                raise NumberFeatureException('Присутствует числовой фактор, который должен быть переведен в категориальный!')
        logger.debug(f'Количество пустых значений в столбцах после заполнения пропусков: \n{product_table.isna().sum()}')
        return product_table

    def encode_features(self, original_product_table: pd.DataFrame):
        """
        Метод кодирования факторов модели
        """
        logger.info('Начинаем кодирование факторов.')
        product_table = original_product_table.copy()
        product_table_encoded = pd.DataFrame()
        for feature in self.input_table_types_dict:
            if self.input_table_types_dict.get(feature) == 'Стр':
                handled_feature = self.handle_text_feature(product_table[feature])
                #handled_feature = handled_feature.fillna('')
                # todo ПРОВЕРИТЬ НА ПРОПУСКИ
            elif self.input_table_types_dict.get(feature) == 'Кат':
                handled_feature = self.handle_cat_feature(product_table[feature])
            else:
                handled_feature = pd.DataFrame(product_table[feature])
            handled_feature.columns = [str(col)+'_'+str(feature) for col in handled_feature.columns]  
            product_table_encoded = pd.concat([product_table_encoded,handled_feature],axis=1)

        # Т.к. мы пересобирали датасет заново во время кодирования => перепроверим на наличие пропущеных значений
        if product_table_encoded.isna().sum().sum()>0:
            logger.error('Появились пустые значения после кодирования переменных')
            raise EmptyValuesAfterEncoding('Появились пустые значения после кодирования переменных!')
        logger.info(f'Получилось {product_table_encoded.columns.size} факторов после кодирования')
        return product_table_encoded

    def handle_cat_feature(self, cat_feature: pd.Series):
        """
        Метод кодирования категориального фактора
        """
        cat_feature = cat_feature.astype(str)
        unique_values_count = cat_feature.drop_duplicates().size
        # todo Убрать комментарий у OneHotEncodingLimit
        if unique_values_count <= 30:#OneHotEncodingLimit:
            # OneHotEncoding
            cat_feature_encoded = pd.get_dummies(cat_feature)
        else:
            # LabelEncoding - чтобы сильно не увеличивать количество факторов
            le = LabelEncoder()
            cat_feature_encoded = pd.DataFrame(le.fit_transform(cat_feature))
        return cat_feature_encoded

    def handle_text_feature(self, text_feature: pd.Series):
        '''
        Метод обработки и кодирования строкового фактора:
        - Проводим препроцессинг
        - Формируем "Мешок строк" (bag of words)
        '''
        processed_text_feature = text_feature_preprocessing(text_feature)
        
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


# Тестируем
#productTable = pd.read_excel('tire_classificator_data.xlsx')

#final_product_table = RawProductsTableHandler(productTable).handle_raw_product_table()
#final_product_table.to_excel('tire_trainset_after_primary_handling.xlsx')#, encoding='cp1251')
# todo ИЗМЕНИТЬ АЛГОРИТМ КОДИРОВАНИЯ ПРИЗНАКОВ