import pandas as pd
from loguru import logger
from config import TargetNameColumn
from src.make_classification import Classifier


def get_formed_trainset(raw_product_table: pd.DataFrame = None):
    trainset_features, trainset_targets = Classifier(product_table).classify_products()
    return trainset_features, trainset_targets


product_table = pd.read_excel('data/paper_classificator_data.xlsx')
trainset_features, trainset_targets = get_formed_trainset(product_table)
