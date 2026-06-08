import pandas as pd
import streamlit as st
import requests
from datetime import date, datetime
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import scoreboardv3

USA_TIMEZONE = "America/New_York"
HKT_TIMEZONE = "Asia/Hong_Kong"


def _to_hkt(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(HKT_TIMEZONE)


def _format_time(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return "-"
    return value.strftime("%Y-%m-%d %H:%M")


def _format_score(score: object) -> str:
    if pd.isna(score):
        return "0"
    return str(int(score)) if float(score).is_integer() else str(score)


def _parse_matchup(game_code: object) -> tuple[str, str]:
    if not isinstance(game_code, str) or len(game_code) < 6:
        return ("", "")
    code = game_code.split("/")[-1]
    if len(code) < 6:
        return ("", "")
    return code[:3], code[3:6]


def predict_winner(home_team_id: object, away_team_id: object) -> str:
    payload = {
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error calling prediction API: {exc}")
        return "Unknown"

    data = response.json()
    print(f"API response: {response.status_code} - {data}")
    return data.get("predicted_winner_abbr", "Unknown")


@st.cache_data(ttl=300)
def load_latest_mlflow_metrics() -> dict[str, object]:
    base_url = "http://localhost:5001/api/2.0/mlflow"
    experiment_name = "NBA Matchup Prediction"

    experiment_resp = requests.get(
        f"{base_url}/experiments/get-by-name",
        params={"experiment_name": experiment_name},
        timeout=15,
    )
    experiment_resp.raise_for_status()
    experiment = experiment_resp.json().get("experiment")
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    experiment_id = experiment["experiment_id"]
    runs_resp = requests.post(
        f"{base_url}/runs/search",
        json={
            "experiment_ids": [experiment_id],
            "max_results": 1,
            "order_by": ["attributes.start_time DESC"],
        },
        timeout=15,
    )
    runs_resp.raise_for_status()
    runs = runs_resp.json().get("runs", [])
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    latest_run = runs[0]
    metrics_list = latest_run.get("data", {}).get("metrics", [])
    metrics = {
        str(metric.get("key", "")).strip().lower().replace(" ", "_"): metric.get("value")
        for metric in metrics_list
        if isinstance(metric, dict)
    }
    info = latest_run.get("info", {})
    return {
        "experiment_name": experiment_name,
        "run_id": info.get("run_id", "-"),
        "run_name": info.get("run_name", "-"),
        "start_time": info.get("start_time"),
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "roc_auc": metrics.get("roc_auc"),
    }


def get_today_us_date() -> date:
    return pd.Timestamp.now(tz=USA_TIMEZONE).date()


@st.cache_data(ttl=60)
def load_today_games(game_date: str) -> pd.DataFrame:
    scoreboard = scoreboardv3.ScoreboardV3(game_date=game_date)
    _, games_df, teams_df = scoreboard.get_data_frames()[:3]

    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "MATCHUP",
                "Game Time (HKT)",
                "Status",
                "Score",
                "Away Team",
                "Home Team",
            ]
        )

    rows = []
    teams_by_game = {
        game_id: frame.copy()
        for game_id, frame in teams_df.groupby("gameId", sort=False)
    }

    for _, game in games_df.iterrows():
        game_id = game["gameId"]
        away_tricode, home_tricode = _parse_matchup(game.get("gameCode"))
        game_teams = teams_by_game.get(game_id, pd.DataFrame())

        away_team = game_teams[game_teams["teamTricode"] == away_tricode].head(1)
        home_team = game_teams[game_teams["teamTricode"] == home_tricode].head(1)

        if away_team.empty or home_team.empty:
            ordered = list(game_teams.to_dict("records"))
            away_record = ordered[0] if len(ordered) > 0 else {}
            home_record = ordered[1] if len(ordered) > 1 else {}
        else:
            away_record = away_team.iloc[0].to_dict()
            home_record = home_team.iloc[0].to_dict()

        game_time_hkt = _to_hkt(game.get("gameTimeUTC"))
        away_score = _format_score(away_record.get("score"))
        home_score = _format_score(home_record.get("score"))
        predicted_winner = predict_winner(
            home_record.get("teamId"),
            away_record.get("teamId"),
        )

        rows.append(
            {
                "MATCHUP": f"{away_record.get('teamTricode', away_tricode)} @ {home_record.get('teamTricode', home_tricode)}",
                "Game Time (HKT)": _format_time(game_time_hkt),
                "Status": game.get("gameStatusText", "-"),
                "Score": f"{away_score} - {home_score}",
                "Away Team": f"{away_record.get('teamCity', '')} {away_record.get('teamName', '')}".strip(),
                "Home Team": f"{home_record.get('teamCity', '')} {home_record.get('teamName', '')}".strip(),
                "Predicted Winner": predicted_winner,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="NBA Matchups", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1400px; }
        h1 { font-size: 2.5rem; }
        h2, h3 { font-size: 1.6rem; }
        .stDataFrame { font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("NBA Matchups")

    tab_games, tab_results = st.tabs(["Today's Games","Model Results"])

   

    with tab_games:
        today_hkt = datetime.now(ZoneInfo(HKT_TIMEZONE)).date()
        today_us = get_today_us_date()
        today_games = load_today_games(today_us.isoformat())

        if today_games.empty:
            st.info("No NBA games today.")
        else:
            st.caption(f"Today's games in HKT: {today_hkt}")
            st.dataframe(
                today_games,
                use_container_width=True,
                hide_index=True,
            )
    with tab_results:
        try:
            mlflow_metrics = load_latest_mlflow_metrics()
            st.subheader("Latest XGBoost training metrics")
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "Accuracy",
                f"{mlflow_metrics['accuracy']:.4f}"
                if mlflow_metrics["accuracy"] is not None
                else "-",
            )
            metric_cols[1].metric(
                "Log loss",
                f"{mlflow_metrics['log_loss']:.4f}"
                if mlflow_metrics["log_loss"] is not None
                else "-",
            )
            metric_cols[2].metric(
                "ROC AUC",
                f"{mlflow_metrics['roc_auc']:.4f}"
                if mlflow_metrics["roc_auc"] is not None
                else "-",
            )
        except (requests.RequestException, ValueError) as exc:
            st.info(f"MLflow metrics unavailable: {exc}")

if __name__ == "__main__":
    main()
