import os
import re
from functools import lru_cache
from dotenv import load_dotenv
import awswrangler as wr
import boto3
load_dotenv()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "NBA Matchup Prediction")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/latest.json")
ATHENA_DB = os.environ.get("ATHENA_DB", "nbadata")
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP")
ATHENA_OUTPUT = os.environ.get("ATHENA_OUTPUT", "")
TRAINING_TABLE = os.environ.get("TRAINING_TABLE", "game_features_history")
INFERENCE_TABLE = os.environ.get("INFERENCE_TABLE", "feature_inference")
WINPCT_TABLE = os.environ.get("WINPCT_TABLE", "feature_team_opponent_winpct")

FEATURE_COLS = [
    "HOME",
    "REST_DAYS",
    "BACK_TO_BACK",
    "ROLL_WIN_PCT_5",
    "ROLL_WIN_PCT_10",
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
]


def normalize_metric_name(name: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def normalize_metric_dict(metrics: dict[str, object]) -> dict[str, object]:
    return {normalize_metric_name(key): value for key, value in metrics.items()}


@lru_cache(maxsize=1)
def get_boto3_session() -> boto3.Session:
    return boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    )


def read_athena_query(sql: str):
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS_REGION is required to query Athena.")
    if not ATHENA_OUTPUT:
        raise ValueError("ATHENA_OUTPUT is required to query Athena.")

    return wr.athena.read_sql_query(
        sql=sql,
        database=ATHENA_DB,
        s3_output=ATHENA_OUTPUT,
        workgroup=ATHENA_WORKGROUP,
        ctas_approach=False,
        boto3_session=get_boto3_session(),
    )
