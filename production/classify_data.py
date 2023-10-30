import pandas as pd
from loguru import logger
from src.make_classification import Classifier


def predict_classes(raw_product_table: pd.DataFrame = None):
    predicted_classes = Classifier(raw_product_table).classify_products()
    #predicted_classes = Classifier(raw_product_table).check_accuracy()
    return predicted_classes


#product_table = pd.read_excel('production/data/paper_classificator_data.xlsx')
product_table = pd.read_csv('production/data/tubes_classificator_data.csv', sep=';', encoding='cp1251')
#product_table = pd.read_excel('production/data/tire_classificator_data.xlsx')
#product_table = pd.read_excel('production/data/tubes_classificator_data.xlsx')
print(product_table.columns)
#exit()
logger.debug(f'\n{predict_classes(product_table)}')
