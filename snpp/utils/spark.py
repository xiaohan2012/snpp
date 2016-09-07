from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("Signed Network Experiment")\
        .getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir('.checkpoint')  # stackoverflow errors
