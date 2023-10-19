from abc import ABC
from abc import abstractmethod


class ClassificatorModelInterface(ABC):
    '''
    Абстрактный класс для класса модели классификатора:
        * params_tuner - подбор гиперпараметров (optuna)
        * Обучение модели
        * Оценка качества
        * Формирование итогового предсказание
    '''