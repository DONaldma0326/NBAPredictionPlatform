import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, lit, to_date
## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw CSV data
raw_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("s3://nbaanalysisproject/bronze/raw/Games/")

# Data Cleaning & Transformation
clean_df = raw_df \
    .withColumn("GAME_DATE", to_date(col("GAME_DATE"))) \
    .withColumn("SEASON", col("SEASON").cast("string")) \
    .withColumn("HOME", when(col("MATCHUP").rlike("(?i)vs\\."), lit(1)).otherwise(lit(0))) \
    .withColumn("TEAM_WIN", when(col("WL") == "W", lit(1)).otherwise(lit(0))) \

# Write to Delta Lake (Silver)
clean_df.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("SEASON") \
    .save("s3://nbaanalysisproject/silver/Games/")

# Register as Glue Catalog Table
spark.sql("""
    CREATE TABLE IF NOT EXISTS nbadata.games
    USING DELTA
    LOCATION 's3://nbaanalysisproject/silver/Games/'
""")

print("✅ Silver Delta Lake table updated successfully!")

job.commit()