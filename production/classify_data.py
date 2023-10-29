import pandas as pd
from loguru import logger
from config import TargetColumnName
from src.make_classification import Classifier


def predict_classes(raw_product_table: pd.DataFrame = None):
    predicted_classes = Classifier(raw_product_table).classify_products()
    return predicted_classes


product_table = pd.read_excel('production/data/paper_classificator_data.xlsx')
#product_table = pd.read_excel('data/tire_classificator_data.xlsx')
logger.debug(f'\n{predict_classes(product_table).head()}')
