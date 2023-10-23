import pandas as pd
from loguru import logger
from config import TargetNameColumn
from src.make_classification import Classifier


def classify_data(raw_product_table: pd.DataFrame = None):
    classified_products = Classifier(product_table).classify_products()
    # logger.debug(f"Колонки на выходе: {classifier.columns}")
    # logger.debug(f"Итоговый столбец с предсказанием: {classifier['predicted_class']}")
    return classified_products


product_table = pd.read_excel('data/paper_classificator_data.xlsx')
classifier_output = classify_data(product_table)
logger.debug(classifier_output)
