from nltk.corpus import stopwords
from pymystem3 import Mystem


def text_feature_preprocessing(text_feature, ):
        '''
        Функция преобразования текстовых факторов
        - Переводим в нижний регистр
        - Удаляем знаки препинания
        - Удаляем стоп слова
        - Проводим стеммизацию
        '''
        russian_stopwords = stopwords.words("russian")
        stemmer = SnowballStemmer("russian")
        # TODO ДОПИЛИТЬ
        vectorizer = TfidfVectorizer(
            min_df = 20,
            analyzer='word',
            stop_words = russian_stopwords
        )
        processed_feature = []
        text_feature = text_feature.replace(r'[^\w\s]',' ', regex=True).replace(r'\s+',' ', regex=True).str.lower()
        processed_text_feature = text_feature.apply(text_feature_by_token_processing(stemmer))
        #return processed_text_feature
        return text_feature
    
def text_feature_by_token_processing(text, stemmer, russian_stopwords):
    # TODO ДОПИЛИТЬ
    tokens = [stemmer.stem(token) for token in text.split() if token not in russian_stopwords and token != " "]
    tokens = ' '.join(tokens)
    return tokens