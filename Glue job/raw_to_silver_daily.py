import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, lit, to_date
from delta.tables import DeltaTable
## @params: [JOB_NAME, INGESTION_DATE]
args = getResolvedOptions(sys.argv, ['JOB_NAME','INGEST_DATE'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw CSV data
raw_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f"s3://nbaanalysisproject/bronze/raw/Games/ingest_date={args['INGEST_DATE']}")

# Data Cleaning & Transformation
clean_df = raw_df \
    .withColumn("GAME_DATE", to_date(col("GAME_DATE"))) \
    .withColumn("SEASON", col("SEASON").cast("string")) \
    .withColumn("HOME", when(col("MATCHUP").rlike("(?i)vs\\."), lit(1)).otherwise(lit(0))) \
    .withColumn("TEAM_WIN", when(col("WL") == "W", lit(1)).otherwise(lit(0))) \
    .withColumn("INGEST_DATE", lit(args["INGEST_DATE"])) \
    .dropDuplicates(["GAME_ID", "TEAM_ID"])
# Upsert to Delta Lake (Silver) using Spark DataFrames only.
games_path = "s3://nbaanalysisproject/silver/Games/"
if not DeltaTable.isDeltaTable(spark, games_path):
    raise ValueError(f"Delta table does not exist at {games_path}. Please run the initial load first.")
delta_table = spark.read.format("delta").load(games_path)
silver_table = DeltaTable.forPath(spark, games_path)
(
        silver_table.alias("target")
        .merge(
            clean_df.alias("source"),
            "target.GAME_ID = source.GAME_ID AND target.TEAM_ID = source.TEAM_ID",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
)


print("✅ Silver Delta Lake table updated successfully!")

job.commit()


