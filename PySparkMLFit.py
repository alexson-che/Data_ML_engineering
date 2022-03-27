import io
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    train_data = "result/train/train.parquet"
    #test_data - путь к файлу с данными для оценки качества модели
    test_data = "result/test/test.parquet"
    spark = SparkSession.builder.appName('PySparkTasks').getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "GMT+3")
    ctr_df_train = spark.read.parquet(train_data)
    ctr_df_test = spark.read.parquet(test_data)

    model_results = {}
    feature_train = VectorAssembler(inputCols=ctr_df_train.columns[0:6],outputCol="features")
    feature_vector_train= feature_train.transform(ctr_df_train)

    feature_test = VectorAssembler(inputCols=ctr_df_test.columns[0:6],outputCol="features")
    feature_vector_test = feature_test.transform(ctr_df_test)
    
    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="features", labelCol="ctr")
    # Find best params/model


    pipeline_rf = Pipeline(stages=[feature_train, rf])

    paramGrid = ParamGridBuilder()\
        .addGrid(rf.impurity, ['variance']) \
        .addGrid(rf.maxDepth, [2, 3, 5, 7, 10])\
        .build()

    tvs = TrainValidationSplit(estimator=pipeline_rf,
                               estimatorParamMaps=paramGrid,
                               evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse"),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    # Train model
    rf_model = tvs.fit(ctr_df_train)
    #Make predictions
    predictions = rf_model.transform(ctr_df_test)\

    #Model evaluation on test data
    evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse")
    rmse_rf = evaluator.evaluate(predictions)
    model_results[rf_model] = rmse_rf
    
    
    
    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="features", labelCol="ctr", maxIter=10)
    # Find best params/model

    pipeline_gbt = Pipeline(stages=[feature_train, gbt])

    paramGrid = ParamGridBuilder()\
        .addGrid(gbt.impurity, ['variance']) \
        .addGrid(gbt.lossType, ['squared', 'absolute'])\
        .addGrid(gbt.maxDepth, [2, 5, 7, 10])\
        .addGrid(gbt.maxIter, [10, 15, 20])\
        .build()

    tvs = TrainValidationSplit(estimator=pipeline_gbt,
                               estimatorParamMaps=paramGrid,
                               evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse"),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)
    # Train model.
    gbt_model = tvs.fit(ctr_df_train)


    # Make predictions.
    predictions = gbt_model.transform(ctr_df_test)\
    # Model evaluation on test data
    evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse")
    rmse_gbt = evaluator.evaluate(predictions)
    model_results[gbt_model] = rmse_gbt


    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="ctr")

    pipeline_dt = Pipeline(stages=[feature_train, dt])

    # Find best params/model
    paramGrid = ParamGridBuilder()\
        .addGrid(dt.impurity, ['variance']) \
        .addGrid(dt.maxDepth, [2, 3, 5, 7, 10])\
        .build()

    tvs = TrainValidationSplit(estimator=pipeline_dt,
                               estimatorParamMaps=paramGrid,
                               evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse"),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    # Train model.
    dt_model = tvs.fit(ctr_df_train)

    # Make predictions.
    predictions = dt_model.transform(ctr_df_test)\
    # Model evaluation on test data
    evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse")
    rmse_dt = evaluator.evaluate(predictions)
    model_results[dt_model] = rmse_dt
    
    
    
    
    # select the best model with the lowest RMSE
    a = sorted(model_results.values(), reverse=False)
    for k, v in model_results.items():
        if v == a[0]:
            k
            
    # save the best model
    k.write().overwrite().save('best_model')
    
def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)