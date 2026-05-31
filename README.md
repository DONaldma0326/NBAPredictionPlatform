# 🏀 End-to-End NBA Analytics Data Warehouse

A **fault‑tolerant**, medallion‑architecture data platform that ingests NBA team and player statistics, processes them through AWS S3 and AWS Glue, and serves interactive dashboards via a containerized metabase. Orchestration is handled by **Apache Airflow** (Dockerized), data transformations run on **AWS Glue** (Spark SQL), and a **Metabase** (Dockerized) for Visualize.

---

## 📌 Overview

This project demonstrates a **hybrid data platform**:
- **Data Lake** on AWS S3 with Madellion architecture.
- **Serverless processing** with AWS Glue .
- **Orchestration** via Airflow running in Docker locally for scheduling, monitoring, and failure handling.
- **Interactive dashboards** built with AWS Athena connected to Metabase that run in local.

**Key Features**
- Medallion architecture (Bronze/Silver/Gold) for data quality and governance.
- Fault‑tolerant orchestration with Airflow (retries, alerting, idempotency).
- Containerized control plane – Airflow
- Interactive BI Dashboard - Metabase

---

## 🏗️ Architecture

![alt text](Image/image-1.png)

├──► Kaggle Dataset

├──►  Airflow (Docker) 

├──► S3 (Bronze) - raw data landing zone

├──► AWS Glue (Silver) - clean, deduplicate, type cast

├──► S3 (Silver) - cleaned, partitioned Parquet

├──► AWS Glue (Gold) - build aggregated tables

├──► S3 (Gold) - reporting‑ready datasets

└──► Metabase - dashboards & ad‑hoc analysis



---

## 🛠️ Technologies Used

| Component           | Technology                                    | Purpose                                      |
| ------------------- | --------------------------------------------- | -------------------------------------------- |
| Orchestration       | Apache Airflow (Docker container)             | Schedule and monitor monthly pipeline runs   |
| Data Lake Storage   | Amazon S3                                     | Tiered storage: `bronze/`, `silver/`, `gold/`|
| Data Processing     | AWS Glue (Spark SQL, PySpark)                 | Transform raw data into clean, aggregated tables |
|  Query Engine | AWS Athena          | Store table schemas, enable SQL queries on S3|
| Visualization       | Metabase       | Build interactive dashboards                  |
| Source Data         | Kaggle                                    | Fetch NBA stats    |
| Container Runtime   | Docker / Docker Compose                        | Run Airflow and Metabase locally  |

---

## 📦 Pipeline Details

### 1️⃣ Data Ingestion (Bronze)
- **Airflow DAG** (Docker) triggers monthly (configurable via `schedule_interval`).
- Uses **Kaggle API** to download the latest NBA dataset (CSV/JSON).
- Stores raw files into `s3://your-bucket/bronze/` with ingest date by `year` and `month` (e.g., `year=2024/month=02/`).

### 2️⃣ Data Cleansing (Silver)
- **AWS Glue ETL job** (PySpark script) reads from Bronze.
- Performs:
  - Schema enforcement and type casting (e.g., string → integer, date).
  - Handling missing values (drop or impute).
  - Deduplication based on game/player IDs.
  - Handle traded mid season
- Writes **Parquet** format (optimized for analytics) to `s3://your-bucket/silver/`, partitioned by `season` and `team`.


### 3️⃣ Data Aggregation (Gold)
- **Second Glue job** builds dimensional models:
  - **Player career stats** – averages per season, shooting percentages.
  - **Team performance** – win/loss streaks, offensive/defensive ratings.
- Outputs to `s3://your-bucket/gold/` in Parquet

### 4️⃣ Feature Store
- **Daily Glue job** reads the Silver games Delta table and builds two feature outputs:
  - **Historical game feature table** – one row per team-game with rolling pregame features and the final result label.
  - **Inference snapshot table** – current team-state features that can be joined with a live schedule before prediction.
- Both tables update daily, with the historical table upserting only the new game rows.

### 5️⃣ Visualization
- BI tool connects to the AWS athena
- Users can run SQL queries directly on Glue Data catalog
- Dashboard with interactive filter, Player Comparison and Team trend

---

## ⚙️ Fault Tolerance & Reliability

- **Airflow retries** – failed tasks automatically retry up to 3 times (configurable in DAG).
- **Data validation** – Glue jobs perform row count and schema checks before writing.
- **Idempotent writes** – each run overwrites only the relevant partitions, avoiding duplicates. 
- **S3 versioning** – enabled on Bronze bucket to recover raw data if needed.

---


### Prerequisites
- AWS account with permissions for S3, Glue, and Athena.
- Docker installed.
- Python installed

### To start

Create an AWS s3 bucket, Create bronze, silver, gold folder

Create an IAM or Use an Existing Role Grant AmazonS3FullAccess and AWSGlueConsoleFullAccess
create access key
Keep the access key id and secret

```bash
cd ./Airflow

docker-compose up -d
```
This launches:
Airflow webserver & scheduler (UI at http://localhost:8080)
![alt text](Image/image-2.png)

Sign in and create connection
![alt text](Image/image-3.png)

![alt text](Image/image-4.png)
insert the key and secret you got in step2



```bash

docker run -d -p 3000:3000 --name metabase metabase/metabase 
```
This lauches:
metabase dashboard (UI at http://localhost:3000)

![alt text](Image/image-5.png)
**Sign up may require when first launch


![alt text](Image/image-6.png)
Select Amazon Athena
Go to the admin portal and select create a database
Fill the required information
![alt text](Image/image-7.png)
The database connection is established and can be used to build dashboard now !!!

📊 Example Dashboard

Player Comparison: Top 10 player with highest 3pt percentage

Leaderboards: Top 10 scorers since 2016.

Team Trends: Line chart of points per game over the last 10 seasons.



![alt text](Image/image.png)


You are an experience software engineer.
I am trying to build a system that predict the NBA game out come.
With historical data, realtime data.
for historical data i will build a postgres datawarehouse. That i am going to load the
for realtime data i will use nba_api.live.nba.endpoints
The logic flow of the model should be



AWS setup


Get a first time setup
fault torlance idenpont , airflow
Bronze every day
xxxx-yyyy_mm_dd.csv


glue
Silver
delta lake extracted from Bronze
1:1 Games playbyplay Player Team

glue
Gold
Metrics

Athena
Metabase


open an EC2 install Python, Docker and the needed package


download all season data and upload to S3
create medaliaz structure
bronze/raw/players
bronze/raw/games
bronze/raw/team

Glue (init load)

silver/players (Delta Lake)
silver/games (Delta Lake)
silver/team (Delta Lake)


(run some init transformation to deduplicate and suitable for SCD2)

install nba_api and pandas in airflow-dag-processor and airflow-worker


--conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog --conf spark.delta.logStore.class=org.apache.spark.sql.delta.storage.S3SingleDriverLogStore