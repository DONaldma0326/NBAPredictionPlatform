from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook  
import requests
import zipfile
import os
from pathlib import Path
extract_dir = '/tmp/extracted'
# Default DAG args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'schedule_interval': '@weekly',
}

# DAG definition
dag = DAG(
    'kaggle_nba_to_S3',
    default_args=default_args,
    description='Download Kaggle NBA dataset using requests and upload to S3',
    catchup=False,
    tags=['kaggle', 'nba', 'project'],
)

# Task 1: Download from Kaggle API using requests
def download_kaggle_dataset(**kwargs):
    url = f"https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats"
    
    download_path = '/tmp/NBA_download.zip'
    
    # Download with Basic Auth
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {download_path}")
    print(f"Download response status code: {response.status_code}")
    print(f"Downloaded file size: {os.path.getsize(download_path)} bytes")
    return download_path  

download_data_task = PythonOperator(
    task_id='download_kaggle_dataset',
    python_callable=download_kaggle_dataset,
    dag=dag,    
)

def remove_useless_file(**kwargs):
     #delete the nba.sqlite, which is not needed
    useless_file = ['All-Star Selections.csv','Draft Pick History.csv','Team Abbrev.csv','Player Award Shares.csv','End of Season Teams (Voting).csv','End of Season Teams.csv']
   
    for file in useless_file:
        file_path = os.path.join(extract_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed unnecessary file {file_path}")

remove_useless_files_task = PythonOperator(
    task_id='remove_useless_files',
    python_callable=remove_useless_file,
    dag=dag,
)

# Task 2: Unzip and Upload to S3 (same as before)
def upload_to_S3(**kwargs):
    zip_path = kwargs['ti'].xcom_pull(task_ids='download_kaggle_dataset')

    print(f"Extracting {zip_path} to {extract_dir}")


    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
   
    hook = S3Hook(aws_conn_id='S3')
    bucket = 'nbaanalysisproject'  # Change to your bucket
    
    # Upload each file
    for root, _, files in os.walk(extract_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = f'bronze/stats/ingest_date={datetime.now().strftime("%Y-%m")}/{file}' 

            if hook.check_for_key(key=s3_key, bucket_name=bucket):
                hook.delete_objects(bucket=bucket, keys=[s3_key])
                print(f"Deleted existing file {s3_key} from bucket {bucket}")

            hook.load_file(filename=local_path, key=s3_key, bucket_name=bucket, replace=True)
            print(f"Uploaded {file} to {bucket}/{s3_key}")
    
    # Cleanup
    os.remove(zip_path)
    for f in Path(extract_dir).rglob('*'):
        if f.is_file():
            f.unlink()

upload_task = PythonOperator(
    task_id='upload_to_S3',
    python_callable=upload_to_S3,
    dag=dag,
)

# Task dependencies
download_data_task >> remove_useless_files_task >> upload_task