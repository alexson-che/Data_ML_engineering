# Data_ML_engineering
 Pyspark project, consists of several pyspark jobs, that collect marketing campaign data from the parquet file, prepare the dataset and train a several ML models to chose the best one and save it for further implementation.


## Requirements
* Docker
* Jupyter Notebook
* PySpark (3.1.1)



### PySparkJob.py
Загружает исходный файл, обрабатывает его, преобразует к целевому формату и с помощью метода [randomsplit](https://spark.apache.org/docs/2.4.1/api/python/pyspark.sql.html?highlight=split#pyspark.sql.DataFrame.randomSplit) делит исходную выборку на тренировочную и тестовую в соотношении train/test = 0.75/0.25

### PySparkMLFit.py
Задача, которая тренирует модели [RandomForestRegressor](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-regression), [GBTRegressor](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression), [DecisionTreeRegressor](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression), подбирает оптимальные гиперпараметры с помощью [Train-Validation Split](https://spark.apache.org/docs/3.2.1/ml-tuning.html#train-validation-split) производит на базе test датасета оценку качества моделей с помощью RegressionEvaluator и ошибки RMSE модели, с последующим сохранением наилучшей модели для последующего использования.



### PySparkMLPredict.py
Задача, которая загружает модель и строит предсказание целевой фичи (CTR) над переданными ей данными
