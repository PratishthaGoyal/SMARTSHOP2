# backend/ml_api.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import sqlite3
from backend.ml_model import run_ml_models

app = FastAPI(title="SmartShop ML API")
DB_PATH = "database/my_database.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM my_table", conn)
    conn.close()
    return df


@app.get("/predict")
def predict():
    """Run ML models and return JSON predictions"""
    df = load_data()
    if df.empty:
        return JSONResponse({"error": "No data found in database."}, status_code=404)

    results = run_ml_models(df)
    return results
