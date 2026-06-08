import os
from functools import lru_cache
from typing import Dict

import awswrangler as wr
import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/latest.json")
ATHENA_DB = os.environ.get("ATHENA_DB", "nbadata")
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP")
ATHENA_OUTPUT = os.environ.get("ATHENA_OUTPUT","")
INFERENCE_TABLE = os.environ.get("INFERENCE_TABLE", "feature_inference")
WINPCT_TABLE = os.environ.get("WINPCT_TABLE", "feature_team_opponent_winpct")
TEAM_ID_COL = os.environ.get("TEAM_ID_COL", "team_id")

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

HOME_REQUIRED = [
    TEAM_ID_COL,
    "game_date",
    "last_game_date",
    "roll_win_pct_5",
    "roll_win_pct_10",
    "roll_fg_pct_10",
    "roll_efg_pct_10",
    "roll_ast_to_ratio_10",
    "roll_reb_10",
    "roll_stl_10",
    "roll_blk_10",
    "roll_tov_10",
]

AWAY_REQUIRED = [
    TEAM_ID_COL,
    "roll_win_pct_10",
    "roll_fg_pct_10",
    "roll_reb_10",
    "roll_stl_10",
    "roll_blk_10",
    "roll_tov_10",
]

app = FastAPI(title="NBA Game Prediction API")

class PredictRequest(BaseModel):
    home_team_id: int = Field(..., description="Home team ID, e.g. 1610612747")
    away_team_id: int = Field(..., description="Away team ID, e.g. 1610612738")


def load_model() -> XGBClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Set MODEL_PATH env var."
        )
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def _read_athena(sql: str) -> pd.DataFrame:
    if not REGION:
        raise ValueError("AWS_REGION is required to query Athena.")
    if not ATHENA_OUTPUT:
        raise ValueError("ATHENA_OUTPUT is required to query Athena.")
    session = boto3.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"), region_name=REGION)
    return wr.athena.read_sql_query(
        sql=sql,
        database=ATHENA_DB,
        s3_output=ATHENA_OUTPUT,
        workgroup=ATHENA_WORKGROUP,
        ctas_approach=False,
        boto3_session=session,
   
    )


@lru_cache(maxsize=1)
def load_feature_store() -> pd.DataFrame:
    sql = f"SELECT * FROM {ATHENA_DB}.{INFERENCE_TABLE}"
    store = _read_athena(sql)
    if TEAM_ID_COL not in store.columns:
        raise ValueError(f"Missing column {TEAM_ID_COL} in inference table.")
    store["game_date"] = pd.to_datetime(store["game_date"], errors="coerce")
    store["last_game_date"] = pd.to_datetime(store["last_game_date"], errors="coerce")
    return store


@lru_cache(maxsize=1)
def load_winpct(home_team_id: int, away_team_id: int) -> float:
    winpct_sql = (
        f"SELECT winpct "
        f"FROM {ATHENA_DB}.{WINPCT_TABLE} "
        f"WHERE team_id = '{home_team_id}' AND opp_id = '{away_team_id}' "
        f"AND season = (SELECT max(season) FROM {ATHENA_DB}.{WINPCT_TABLE})"
    )
    winpct_df = _read_athena(winpct_sql)
    if winpct_df.empty:
        raise ValueError("No win percentage found in winpct table.")
    return float(winpct_df.iloc[0]["winpct"])



def get_latest_team_row(store: pd.DataFrame, team_id: str) -> pd.Series:
    team_rows = store[store['team_id'] == team_id]
    if team_rows.empty:
        raise KeyError(f"Team ID {team_id} not found in feature store.")
    return team_rows.iloc[-1]


def validate_columns(store: pd.DataFrame) -> None:
    required = set(HOME_REQUIRED + AWAY_REQUIRED)
    missing = sorted(required - set(store.columns))
    if missing:
        raise ValueError(f"Feature store missing columns: {', '.join(missing)}")


def get_team_vs_opp_win_pct(
    winpct: pd.DataFrame, team_id: int, opp_id: int
) -> float:
    required = {"team_id", "opp_id", "winpct"}
    missing = required - set(winpct.columns)
    if missing:
        raise ValueError(f"Winpct table missing columns: {', '.join(sorted(missing))}")
    rows = winpct[(winpct["team_id"] == team_id) & (winpct["opp_id"] == opp_id)]
    if rows.empty:
        raise KeyError(
            f"No winpct for team {team_id} vs {opp_id}."
        )
    return float(rows.iloc[-1]["winpct"])


def build_feature_row(
    home_row: pd.Series,
    away_row: pd.Series,
    team_vs_opp_win_pct: float,
) -> Dict[str, float]:
    rest_days = (
        (home_row["game_date"] - home_row["last_game_date"]).days - 1
        if pd.notna(home_row["game_date"]) and pd.notna(home_row["last_game_date"])
        else 0
    )
    back_to_back = 1 if rest_days == 0 else 0
    return {
        "HOME": 1,
        "REST_DAYS": float(rest_days),
        "BACK_TO_BACK": float(back_to_back),
        "ROLL_WIN_PCT_5": float(home_row["roll_win_pct_5"]),
        "ROLL_WIN_PCT_10": float(home_row["roll_win_pct_10"]),
        "ROLL_FG_PCT_10": float(home_row["roll_fg_pct_10"]),
        "ROLL_EFG_PCT_10": float(home_row["roll_efg_pct_10"]),
        "ROLL_AST_TO_RATIO_10": float(home_row["roll_ast_to_ratio_10"]),
        "ROLL_REB_10": float(home_row["roll_reb_10"]),
        "ROLL_STL_10": float(home_row["roll_stl_10"]),
        "ROLL_BLK_10": float(home_row["roll_blk_10"]),
        "ROLL_TOV_10": float(home_row["roll_tov_10"]),
        "OPP_ROLL_WIN_PCT_10": float(away_row["roll_win_pct_10"]),
        "OPP_ROLL_FG_PCT_10": float(away_row["roll_fg_pct_10"]),
        "OPP_ROLL_REB_10": float(away_row["roll_reb_10"]),
        "OPP_ROLL_STL_10": float(away_row["roll_stl_10"]),
        "OPP_ROLL_BLK_10": float(away_row["roll_blk_10"]),
        "OPP_ROLL_TOV_10": float(away_row["roll_tov_10"]),
        "TEAM_VS_OPP_SEASON_WIN_PCT": float(team_vs_opp_win_pct),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, object]:
    try:
        model = load_model()
        store = load_feature_store()
        validate_columns(store)

        home_team_id = payload.home_team_id
        away_team_id = payload.away_team_id

        if home_team_id == away_team_id:
            raise HTTPException(status_code=400, detail="Home and away teams must differ.")
        print(store.head())
        home_row = get_latest_team_row(store, str(home_team_id))
        away_row = get_latest_team_row(store, str(away_team_id))

        team_vs_opp_win_pct = load_winpct(home_team_id, away_team_id)

        
        features = build_feature_row(home_row, away_row, team_vs_opp_win_pct)
        input_df = pd.DataFrame([features], columns=FEATURE_COLS)
        proba = float(model.predict_proba(input_df)[0, 1])

        predicted_winner_id = home_team_id if proba >= 0.5 else away_team_id
        confidence = max(proba, 1 - proba)

        home_abbr = home_row.get("team_abbreviation")
        away_abbr = away_row.get("team_abbreviation")
        predicted_abbr = (
            home_abbr if predicted_winner_id == home_team_id else away_abbr
        )
       
        return {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "predicted_winner_id": predicted_winner_id,
            "home_team_abbr": home_abbr,
            "away_team_abbr": away_abbr,
            "predicted_winner_abbr": predicted_abbr,
            "home_win_probability": proba,
            "confidence": confidence,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
