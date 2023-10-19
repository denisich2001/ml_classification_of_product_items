**Классификация товаров**

ML-алгоритм для классификации товаров

**Структура классов**:  
    * Класс для общего workflow (**Classificator**)  
        * Класс для формирования трейнсета и методов его обработки (**TrainsetHandler**)  
            * Класс для обработки сырых данных (**RawDataHandler**)  
        * Интерфейс для модели (чтобы можно было добавлять другие алгоритмы, не только RandomForest)(**ClassificatorModelInterface**)  
            * Класс для RandomForest (**RandomForestModel**)  