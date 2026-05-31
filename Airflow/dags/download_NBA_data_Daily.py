from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import random
import time
import pandas as pd
from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
from nba_api.stats.endpoints import LeagueGameFinder

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    }


dag = DAG(
    'download_nba_data_daily',
    default_args=default_args,
    schedule='0 6 * * *',
    description='Fetch today\'s NBA games from nba_api, save them as a CSV, and upload to S3',
    catchup=False,
    tags=['nba_api', 'nba', 'project'],
)


def fetch_df(endpoint_cls, **kwargs):
    last_error = None
    for attempt in range(3):
        try:
            endpoint = endpoint_cls(timeout=60, **kwargs)
            return endpoint.get_data_frames()[0]
        except Exception as exc:
            last_error = exc
            wait = 2 + (attempt * 3) + random.uniform(0.5, 1.5)
            print(f"retry {attempt + 1}/3 for {endpoint_cls.__name__}: {exc} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise last_error


def fetch_todays_games_to_s3(**kwargs):
    output_dir = "/tmp/nba_daily_data"
    os.makedirs(output_dir, exist_ok=True)

    pacific = datetime.now(ZoneInfo('America/Los_Angeles'))
    today_str = pacific.strftime('%m/%d/%Y')
    filename_today = pacific.strftime('%Y_%m_%d.csv')
    
    # Determine the current NBA season based on the date
    if pacific.month < 10:
        season = f"{pacific.year - 1}-{str(pacific.year)[2:]}"
    else:
        season = f"{pacific.year}-{str(pacific.year + 1)[2:]}"

    season_types = ['Regular Season', 'Playoffs']
    final_games_df = None

    print(f"Downloading today's data for {today_str} (season {season})")
    for season_type in season_types:
        print(f"Fetching data for {season_type}...")
        games_df = fetch_df(
            LeagueGameFinder,
            date_from_nullable=today_str,
            date_to_nullable=today_str,
            season_type_nullable=season_type,
            league_id_nullable='00',
        )
        games_df['season'] = season
        games_df['season_type'] = season_type
        games_df['ingestion_date'] = pacific.strftime('%Y-%m-%d')
        final_games_df = games_df if final_games_df is None else pd.concat(
            [final_games_df, games_df],
            ignore_index=True,
        )

    if final_games_df is None or final_games_df.empty:
        print(f'No games found for {today_str}; skipping upload.')
        return None

    games_path = os.path.join(output_dir, f'games_{season}_{filename_today}')
    final_games_df.to_csv(games_path, index=False)

    hook = S3Hook(aws_conn_id='S3')
    bucket = 'nbaanalysisproject'
    s3_key = (
        'bronze/raw/Games/'
        f"ingest_date={pacific.strftime('%Y-%m-%d')}/{os.path.basename(games_path)}"
    )

    if hook.check_for_key(key=s3_key, bucket_name=bucket):
        hook.delete_objects(bucket=bucket, keys=[s3_key])
        print(f"Deleted existing file {s3_key} from bucket {bucket}")

    hook.load_file(filename=games_path, key=s3_key, bucket_name=bucket, replace=True)
    print(f"Uploaded games CSV to {bucket}/{s3_key}")

    os.remove(games_path)
    return pacific.strftime('%Y-%m-%d')


fetch_and_upload_task = PythonOperator(
    task_id='download_nba_data_daily',
    python_callable=fetch_todays_games_to_s3,
    dag=dag,
)


def should_trigger_glue_job(**kwargs):
    return kwargs['ti'].xcom_pull(task_ids='download_nba_data_daily') is not None


trigger_glue_gate = ShortCircuitOperator(
    task_id='should_trigger_raw_to_silver_games_daily',
    python_callable=should_trigger_glue_job,
    dag=dag,
)

trigger_silver_glue_job = GlueJobOperator(
    task_id='raw_to_silver_games_daily',
    aws_conn_id='Glue',
    job_name='raw_to_silver_games_Daily',
    region_name='ap-southeast-1',
    script_args={
        '--INGEST_DATE': "{{ ti.xcom_pull(task_ids='download_nba_data_daily') }}",
    },
    wait_for_completion=True,
    deferrable=False,
    dag=dag,
)

trigger_feature_glue_job = GlueJobOperator(
    task_id='silver_to_gold_games_daily',
    aws_conn_id='Glue',
    job_name='silver_to_gold_games_daily',
    region_name='ap-southeast-1',
    script_args={
        '--INGEST_DATE': "{{ ti.xcom_pull(task_ids='download_nba_data_daily') }}",
    },
    wait_for_completion=True,
    deferrable=False,
    dag=dag,
)

fetch_and_upload_task >> trigger_glue_gate >> trigger_silver_glue_job >> trigger_feature_glue_job