import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer


def text_feature_preprocessing(text_feature: pd.Series):
    """
        Функция преобразования текстовых факторов
        - Переводим в нижний регистр
        - Удаляем знаки препинания
        - Удаляем стоп слова
        - Проводим стеммизацию
    """
    text_feature = text_feature.replace(r'[^\w\s]', ' ', regex=True).replace(r'\s+', ' ', regex=True).str.lower()
    processed_text_feature = text_feature.apply(text_feature_by_token_processing)
    return processed_text_feature


def text_feature_by_token_processing(text):
    """
    Проводим стемминг.
    При наличии русских букв в строке - используем Стеммер для русского языка, иначе - для английского
    """
    text = str(text)
    if bool(re.search('[а-яА-Я]', text)):
        stemmer = SnowballStemmer("russian")
    else:
        stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in text.split()]
    tokens = ' '.join(tokens)
    return tokens
