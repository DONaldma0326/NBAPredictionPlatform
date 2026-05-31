import sys
from datetime import datetime

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from delta.tables import DeltaTable
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F


args = getResolvedOptions(sys.argv, ["JOB_NAME", "INGEST_DATE"])
process_date = datetime.strptime(args["INGEST_DATE"], "%Y-%m-%d").date()

SILVER_PATH = "s3://nbaanalysisproject/silver/Games/"
HISTORICAL_FEATURE_PATH = "s3://nbaanalysisproject/gold/features/game_history/"
INFERENCE_FEATURE_PATH = "s3://nbaanalysisproject/gold/features/inference/"
MATCHUP_LOOKUP_PATH = "s3://nbaanalysisproject/gold/features/team_opponent_winpct/"
CATALOG_DATABASE = "nbadata"
HISTORICAL_TABLE = "game_features_history"
INFERENCE_TABLE = "game_features_inference"
MATCHUP_LOOKUP_TABLE = "team_opponent_winpct_lookup"


sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)


def source_column(frame, *candidate_names):
    for candidate_name in candidate_names:
        if candidate_name in frame.columns:
            return F.col(candidate_name)
    raise KeyError(f"None of the columns {candidate_names} exist in the source frame.")


def canonicalize_games(frame):
    canonical_select = [
        source_column(frame, "GAME_ID").cast("string").alias("GAME_ID"),
        F.to_date(source_column(frame, "GAME_DATE")).alias("GAME_DATE"),
        source_column(frame, "SEASON", "season", "SEASON_ID").cast("string").alias("SEASON"),
        source_column(frame, "TEAM_ID").cast("string").alias("TEAM_ID"),
        source_column(frame, "TEAM_ABBREVIATION").cast("string").alias("TEAM_ABBREVIATION"),
        source_column(frame, "TEAM_NAME").cast("string").alias("TEAM_NAME"),
        source_column(frame, "MATCHUP").cast("string").alias("MATCHUP"),
        source_column(frame, "WL").cast("string").alias("WL"),
        source_column(frame, "PTS").cast("double").alias("PTS"),
        source_column(frame, "FGM").cast("double").alias("FGM"),
        source_column(frame, "FGA").cast("double").alias("FGA"),
        source_column(frame, "FG_PCT").cast("double").alias("FG_PCT"),
        source_column(frame, "FG3M").cast("double").alias("FG3M"),
        source_column(frame, "FG3A").cast("double").alias("FG3A"),
        source_column(frame, "FG3_PCT").cast("double").alias("FG3_PCT"),
        source_column(frame, "FTM").cast("double").alias("FTM"),
        source_column(frame, "FTA").cast("double").alias("FTA"),
        source_column(frame, "FT_PCT").cast("double").alias("FT_PCT"),
        source_column(frame, "OREB").cast("double").alias("OREB"),
        source_column(frame, "DREB").cast("double").alias("DREB"),
        source_column(frame, "REB").cast("double").alias("REB"),
        source_column(frame, "AST").cast("double").alias("AST"),
        source_column(frame, "STL").cast("double").alias("STL"),
        source_column(frame, "BLK").cast("double").alias("BLK"),
        source_column(frame, "TOV").cast("double").alias("TOV"),
        source_column(frame, "PLUS_MINUS").cast("double").alias("PLUS_MINUS"),
    ]

    canonical_frame = frame.select(*canonical_select).dropDuplicates(["GAME_ID", "TEAM_ID"])
    canonical_frame = canonical_frame.withColumn("LAST_GAME_DATE", F.col("GAME_DATE"))
    canonical_frame = canonical_frame.withColumn("HOME", F.when(F.col("MATCHUP").rlike("(?i)vs\\."), F.lit(1)).otherwise(F.lit(0)))
    canonical_frame = canonical_frame.withColumn("TEAM_WIN", F.when(F.col("WL") == F.lit("W"), F.lit(1)).otherwise(F.lit(0)))
    canonical_frame = canonical_frame.withColumn(
        "eFG_PCT",
        F.when(F.col("FGA") == 0, F.lit(None)).otherwise((F.col("FGM") + F.lit(0.5) * F.col("FG3M")) / F.col("FGA")),
    )
    canonical_frame = canonical_frame.withColumn(
        "AST_TO_RATIO",
        F.when(F.col("TOV") == 0, F.lit(None)).otherwise(F.col("AST") / F.col("TOV")),
    )
    return canonical_frame


def add_team_history_features(frame):
    team_window = Window.partitionBy("SEASON", "TEAM_ID").orderBy("GAME_DATE", "GAME_ID")

    frame = frame.withColumn("ROLL_WIN_PCT_5", F.avg("TEAM_WIN").over(team_window.rowsBetween(-5, -1)))
    frame = frame.withColumn("ROLL_WIN_PCT_10", F.avg("TEAM_WIN").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("TEAM_SEASON_WIN_PCT", F.avg("TEAM_WIN").over(team_window.rowsBetween(Window.unboundedPreceding, -1)))
    frame = frame.withColumn("ROLL_FG_PCT_10", F.avg("FG_PCT").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_EFG_PCT_10", F.avg("eFG_PCT").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_AST_TO_RATIO_10", F.avg("AST_TO_RATIO").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_REB_10", F.avg("REB").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_STL_10", F.avg("STL").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_BLK_10", F.avg("BLK").over(team_window.rowsBetween(-10, -1)))
    frame = frame.withColumn("ROLL_TOV_10", F.avg("TOV").over(team_window.rowsBetween(-10, -1)))
    return frame


def attach_opponent_features(frame):
    opponent_select = [
        "GAME_ID",
        "SEASON",
        F.col("TEAM_ID").alias("OPP_TEAM_ID"),
        F.col("TEAM_ABBREVIATION").alias("OPP_TEAM_ABBREVIATION"),
        F.col("TEAM_NAME").alias("OPP_TEAM_NAME"),
        F.col("TEAM_WIN").alias("OPP_TEAM_WIN"),
        F.col("ROLL_WIN_PCT_10").alias("ROLL_WIN_PCT_10_OPP"),
        F.col("ROLL_FG_PCT_10").alias("ROLL_FG_PCT_10_OPP"),
        F.col("ROLL_REB_10").alias("ROLL_REB_10_OPP"),
        F.col("ROLL_STL_10").alias("ROLL_STL_10_OPP"),
        F.col("ROLL_BLK_10").alias("ROLL_BLK_10_OPP"),
        F.col("ROLL_TOV_10").alias("ROLL_TOV_10_OPP"),
    ]

    home_games = frame.filter(F.col("HOME") == 1)
    away_games = frame.filter(F.col("HOME") == 0)

    home_view = home_games.join(away_games.select(*opponent_select), on=["GAME_ID", "SEASON"], how="inner")
    away_view = away_games.join(home_games.select(*opponent_select), on=["GAME_ID", "SEASON"], how="inner")

    pair_window = Window.partitionBy("SEASON", "TEAM_ID", "OPP_TEAM_ID").orderBy("GAME_DATE", "GAME_ID")

    def enrich_matchup_view(view_frame):
        return (
            view_frame.withColumn("OPP_ROLL_WIN_PCT_10", F.col("ROLL_WIN_PCT_10_OPP"))
            .withColumn("OPP_ROLL_FG_PCT_10", F.col("ROLL_FG_PCT_10_OPP"))
            .withColumn("OPP_ROLL_REB_10", F.col("ROLL_REB_10_OPP"))
            .withColumn("OPP_ROLL_STL_10", F.col("ROLL_STL_10_OPP"))
            .withColumn("OPP_ROLL_BLK_10", F.col("ROLL_BLK_10_OPP"))
            .withColumn("OPP_ROLL_TOV_10", F.col("ROLL_TOV_10_OPP"))
            .withColumn("WIN", F.col("TEAM_WIN"))
            .withColumn(
                "TEAM_VS_OPP_SEASON_WIN_PCT",
                F.avg("TEAM_WIN").over(pair_window.rowsBetween(Window.unboundedPreceding, -1)),
            )
        )

    return enrich_matchup_view(home_view).unionByName(enrich_matchup_view(away_view))


def build_feature_frame(frame):
    return attach_opponent_features(add_team_history_features(canonicalize_games(frame)))


def latest_team_snapshot(history_frame):
    latest_season_row = history_frame.orderBy(F.col("GAME_DATE").desc(), F.col("GAME_ID").desc()).select("SEASON").first()
    if latest_season_row is None:
        return None

    latest_season = latest_season_row[0]
    latest_team_window = Window.partitionBy("TEAM_ID").orderBy(F.col("GAME_DATE").desc(), F.col("GAME_ID").desc())
    return (
        history_frame.filter(F.col("SEASON") == F.lit(latest_season))
        .withColumn("row_number", F.row_number().over(latest_team_window))
        .filter(F.col("row_number") == 1)
        .drop("row_number")
    )


historical_output_columns = [
    "GAME_DATE",
    "LAST_GAME_DATE",
    "SEASON",
    "GAME_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "MATCHUP",
    "OPP_TEAM_ID",
    "OPP_TEAM_ABBREVIATION",
    "OPP_TEAM_NAME",
    "HOME",
    "WL",
    "TEAM_WIN",
    "WIN",
    "ROLL_WIN_PCT_5",
    "ROLL_WIN_PCT_10",
    "TEAM_SEASON_WIN_PCT",
    "ROLL_FG_PCT_10",
    "ROLL_EFG_PCT_10",
    "ROLL_AST_TO_RATIO_10",
    "ROLL_REB_10",
    "ROLL_STL_10",
    "ROLL_BLK_10",
    "ROLL_TOV_10",
    "OPP_ROLL_WIN_PCT_10",
    "OPP_ROLL_FG_PCT_10",
    "OPP_ROLL_REB_10",
    "OPP_ROLL_STL_10",
    "OPP_ROLL_BLK_10",
    "OPP_ROLL_TOV_10",
    "TEAM_VS_OPP_SEASON_WIN_PCT",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PLUS_MINUS",
    "eFG_PCT",
    "AST_TO_RATIO",
]

inference_output_columns = [
    "GAME_DATE",
    "LAST_GAME_DATE",
    "SEASON",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "ROLL_WIN_PCT_5",
    "ROLL_WIN_PCT_10",
    "ROLL_FG_PCT_10",
    "ROLL_EFG_PCT_10",
    "ROLL_AST_TO_RATIO_10",
    "ROLL_REB_10",
    "ROLL_STL_10",
    "ROLL_BLK_10",
    "ROLL_TOV_10",
]


silver_games_df = spark.read.format("delta").load(SILVER_PATH)

if "INGEST_DATE" in silver_games_df.columns:
    silver_games_df = silver_games_df.filter(F.col("INGEST_DATE") == F.lit(args["INGEST_DATE"]))
elif "ingestion_date" in silver_games_df.columns:
    silver_games_df = silver_games_df.filter(F.col("ingestion_date") == F.lit(args["INGEST_DATE"]))

if silver_games_df.rdd.isEmpty():
    raise ValueError(f"No silver rows found for ingestion date {args['INGEST_DATE']}.")

feature_frame = build_feature_frame(silver_games_df).select(*historical_output_columns)

feature_frame.write.format("delta").mode("overwrite").partitionBy("SEASON").save(HISTORICAL_FEATURE_PATH)

inference_source_df = latest_team_snapshot(feature_frame)

if inference_source_df is not None:
    inference_df = inference_source_df.select(*inference_output_columns)
    inference_df.write.format("delta").mode("overwrite").partitionBy("SEASON").save(INFERENCE_FEATURE_PATH)


matchup_lookup_df = (
    feature_frame.select(
        F.col("SEASON").alias("SEASON"),
        F.col("TEAM_ID").alias("team_id"),
        F.col("OPP_TEAM_ID").alias("opp_id"),
        F.col("TEAM_NAME").alias("team_name"),
        F.col("OPP_TEAM_NAME").alias("opp_name"),
        F.col("TEAM_WIN").alias("team_win"),
    )
    .groupBy("SEASON", "team_id", "opp_id", "team_name", "opp_name")
    .agg(F.avg("team_win").alias("winPCT"))
)

matchup_lookup_df.write.format("delta").mode("overwrite").save(MATCHUP_LOOKUP_PATH)


print(f"Init historical feature rows: {feature_frame.count()}")
print(f"Init inference snapshot rows: {inference_df.count() if inference_source_df is not None else 0}")
print(f"Init matchup lookup rows: {matchup_lookup_df.count()}")

job.commit()