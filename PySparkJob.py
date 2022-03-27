#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, min, max, count, filter
from pyspark.sql import functions as F

def process(spark, input_file, target_path):
	#spark = SparkSession.builder.appName('PySparkJob').getOrCreate() это для запуска из блокнота
	spark.conf.set("spark.sql.session.timeZone", "GMT+3")
	df = spark.read.parquet('clickstream.parquet')

	ndf = df.withColumn('is_cpm', col('ad_cost_type') == 'CPM')
	ndf = ndf.withColumn('is_cpc',col('ad_cost_type') == 'CPC')
	ndf = ndf.groupBy('ad_id','target_audience_count','has_video', 'is_cpm','is_cpc','ad_cost' )     
	.agg(
		min(col('date')).alias('min_date'), 
		max(col('date')).alias('max_date'))
	ndf = ndf.withColumn('day_count', datediff(ndf.max_date, ndf.min_date))
	ndf = ndf.drop('min_date','max_date')

	ndf2 = df.select('ad_id','event').where(col('event')== 'view')
	ndf2 = ndf2.groupBy('ad_id').count()
	ndf2 = ndf2.withColumn('views', col('count'))
	ndf2 = ndf2.drop(col('count'))

	ndf3 = df.select('ad_id','event').where(col('event')== 'click')
	ndf3 = ndf3.groupBy('ad_id').count()
	ndf3 = ndf3.withColumn('clicks', col('count'))
	ndf3 = ndf3.drop(col('count'))
	ndf2 = ndf2.join(ndf3, 'ad_id', 'inner')
	ndf2 = ndf2.withColumn('CTR', col('clicks')/col('views'))
	ndf = ndf.join(ndf2, 'ad_id', 'inner')
	ndf = ndf.selectExpr("cast(ad_id as integer ) ad_id",
                      "cast(target_audience_count as decimal(10,0)) target_audience_count",
                      "cast(has_video as integer) has_video",
                      "cast(is_cpm as integer) is_cpm",
                      "cast(is_cpc as integer) is_cpc",
                      "cast(ad_cost as double ) ad_cost",
                      "cast(day_count as integer) day_count",
                      "cast(CTR as double) ctr",)
	ndf = ndf.coalesce(1)
	splits = ndf.randomSplit([0.75, 0.25])
	splits[0].write.parquet('result/train')
	splits[1].write.parquet('result/test')



def main(argv):
	input_path = argv[0]
	print("Input path to file: " + input_path)
	target_path = argv[1]
	print("Target path: " + target_path)
	spark = _spark_session()
	process(spark, input_path, target_path)


def _spark_session():
	return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
	arg = sys.argv[1:]
	if len(arg) != 2:
		sys.exit("Input and Target path are require.")
	else:
		main(arg)

