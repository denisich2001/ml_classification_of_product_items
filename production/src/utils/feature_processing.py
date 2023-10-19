from nltk.corpus import stopwords
from pymystem3 import Mystem


def text_feature_preprocessing(text_feature):
        '''
        Функция преобразования текстовых факторов
        - Переводим в нижний регистр
        - Удаляем знаки препинания
        - Удаляем стоп слова
        - Проводим лемматизацию
        '''
        russian_stopwords = stopwords.words("russian")
        mystem = Mystem()

        processed_feature = []
        text_feature = text_feature.replace(r'[^\w\s]',' ', regex=True).replace(r'\s+',' ', regex=True).str.lower()
        #processed_text_feature = text_feature.apply(text_feature_by_token_processing(mystem, russian_stopwords))
        #return processed_text_feature
        return text_feature
    
def text_feature_by_token_processing(text, mystem, russian_stopwords):
    tokens = mystem.lemmatize(text)
    tokens = [token for token in tokens if token not in russian_stopwords and token != " "]
    tokens = ' '.join(tokens)
    return tokens