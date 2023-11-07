"""
Ошибки возникающие при классификации данных.
"""


class NoProductsDataException(Exception):
    pass


class NoPredictionsDataException(Exception):
    pass


class UnknownColumnTypeException(Exception):
    pass


class IncorrectColumnNameException(Exception):
    pass


class NumberFeatureException(Exception):
    pass


class EmptyValuesAfterEncoding(Exception):
    pass


class NoTargetColumnException(Exception):
    pass