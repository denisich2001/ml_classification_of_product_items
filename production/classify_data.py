import pandas as pd
from src.make_classification import Classifier


def predict_classes(raw_product_table: pd.DataFrame = None):
    predicted_classes = Classifier(raw_product_table).classify_products()
    return predicted_classes

# To test:
#data = pd.read_excel('data/tire_classificator_data.xlsx', header=0)
#print(predict_classes(data))