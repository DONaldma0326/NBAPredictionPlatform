import pandas as pd
import streamlit as st
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from nba_api.stats.endpoints import scheduleleaguev2

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


@st.cache_data(ttl=6 * 60 * 60)
def load_schedule() -> pd.DataFrame:
    schedule = scheduleleaguev2.ScheduleLeagueV2()
    schedule_df = schedule.get_data_frames()[0]
    date_source = schedule_df.get("gameDateTimeUTC", schedule_df["gameDate"])
    schedule_df["game_datetime_hkt"] = (
        pd.to_datetime(
            date_source,
            errors="coerce",
            infer_datetime_format=True,
            utc=True,
        )
        .dt.tz_convert("Asia/Hong_Kong")
    )
    schedule_df["game_date"] = pd.to_datetime(
        schedule_df["gameDate"],
        errors="coerce",
        infer_datetime_format=True,
    ).dt.date
    print(schedule_df[["game_date"]].head())
    return schedule_df

def predict_winner(home_team: int, away_team: int) -> int:
    payload = {
        "home_team_id": int(home_team),
        "away_team_id": int(away_team),
    }
    response = requests.post("http://localhost:8000/predict", json=payload)
    print(f"API response: {response.status_code} - {response.json()}")
    return response.json().get("predicted_winner_abbr", "Unknown")

st.title("NBA Matchups")

today_hkt = datetime.utcnow().astimezone(ZoneInfo("Asia/Hong_Kong")).date()
week_end = today_hkt + pd.Timedelta(days=7)


games = load_schedule()
week_games = games[(games["game_date"] >= today_hkt) & (games["game_date"] <= week_end)]

if week_games.empty:
        st.info("No games found for the next 7 days.")
else:
        st.caption(f"Showing games from {today_hkt} to {week_end} (HKT)")
        print(week_games[["game_datetime_hkt", "homeTeam_teamId", "homeTeam_teamName", "homeTeam_teamCity", "awayTeam_teamId", "awayTeam_teamName", "awayTeam_teamCity"]].head())
        matchups = week_games[
            [
                "game_datetime_hkt",
                "arenaName",
                "homeTeam_teamId",
                "homeTeam_teamTricode",
                "homeTeam_teamName",
                "homeTeam_teamCity",
                "awayTeam_teamId",
                "awayTeam_teamTricode",
                "awayTeam_teamName",
                "awayTeam_teamCity",
                "homeTeam_score",
                "awayTeam_score",
            ]
        ].copy()
        
        
        matchups["MATCHUP"] = (
            matchups["awayTeam_teamTricode"]
            + " @ "
            + matchups["homeTeam_teamTricode"]
        )
        matchups["Home Team"] = (
            matchups["homeTeam_teamCity"] + " " + matchups["homeTeam_teamName"]
        )
        matchups["Away Team"] = (
            matchups["awayTeam_teamCity"] + " " + matchups["awayTeam_teamName"]
        )
        matchups["HKT Game Time"] = matchups["game_datetime_hkt"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        matchups["Predicted Winner"] = matchups["Home Team"]
        matchups["Predicted Winner"] = matchups.apply(lambda row: predict_winner(row["homeTeam_teamId"], row["awayTeam_teamId"]), axis=1)
        display_cols = [
            "MATCHUP",
            "HKT Game Time",
            "Home Team",
            "Away Team",
            "awayTeam_score",
            "homeTeam_score",
            "arenaName",
            "Predicted Winner",
            "Actual Winner",  
        ]

        coming_games = matchups[week_games["game_date"] > today_hkt]
        coming_games["Actual Winner"] = "TBD"
        today_games = matchups[week_games["game_date"] == today_hkt]
        today_games["Actual Winner"] = today_games.apply(
            lambda row: row["Home Team"] if row["homeTeam_score"] > row["awayTeam_score"] else row["Away Team"] if row["awayTeam_score"] > row["homeTeam_score"] else "TIE",
            axis=1,
        )
        
        if not today_games.empty:
            st.subheader("Today")
            st.dataframe(
                today_games[display_cols],
                use_container_width=True,
                hide_index=True,
            )

        if not coming_games.empty:
            st.subheader("Coming")
            st.dataframe(
                coming_games[display_cols],
                use_container_width=True,
                hide_index=True,
            )

        # matchups = games[["GAME_ID", "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]].copy()
        # matchups["MATCHUP"] = (
        #     matchups["VISITOR_TEAM_ABBREVIATION"]
        #     + " @ "
        #     + matchups["HOME_TEAM_ABBREVIATION"]
        # )
        # st.write(matchups[["MATCHUP"]])
