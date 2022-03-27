import io
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml.tuning import TrainValidationSplitModel

def process(spark, input_file, output_file):
    #input_file - путь к файлу с данными для которых нужно предсказать ctr
    input_path = "result/test/test.parquet"
    ctr_df_test = spark.read.parquet(input_path)
    #output_file - путь по которому нужно сохранить файл с результатами [ads_id, prediction]
    output_file = "ctr_forecast/forecast"
    #Ваш код
    model = TrainValidationSplitModel.read().load('best_model')
    #model = PipelineModel.load("best_model")
    prediction = model.transform(ctr_df_test)
    prediction.write.parquet(output_file)
    
def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)