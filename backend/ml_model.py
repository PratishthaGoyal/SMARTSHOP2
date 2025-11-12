# backend/ml_model.py

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_ml_models(df):
    """Run both regression (forecasting) and classification models on transaction data."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["product_id_code"] = df["product_id"].astype("category").cat.codes
    selected_date = datetime.now()

    results = {"forecast": [], "classification": {}}

    # === 1. Regression (Sales Forecasting) ===
    for prod in df["product_name"].unique():
        prod_data = df[df["product_name"] == prod]
        if len(prod_data) >= 5:
            X = prod_data[["product_id_code", "month", "day_of_month"]]
            y = prod_data["stock_sold"]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            next_month = (selected_date.month % 12) + 1
            pred = model.predict([[prod_data["product_id_code"].iloc[0], next_month, 15]])

            results["forecast"].append({
                "product": prod,
                "predicted_sales": int(pred[0] * 30)
            })

    # === 2. Classification (Performance) ===
    df_class = df[df["stock_sold"] > 0].copy()

    if len(df_class) >= 15:
        df_class["sales_performance"] = pd.cut(
            df_class["stock_sold"],
            bins=[
                0,
                df_class["stock_sold"].quantile(0.33),
                df_class["stock_sold"].quantile(0.66),
                df_class["stock_sold"].max(),
            ],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        X_class = df_class[["product_id_code", "month", "day_of_month", "discount_percent"]]
        y_class = df_class["sales_performance"]

        X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        importances = dict(zip(X_class.columns, clf.feature_importances_))
        results["classification"] = {
            "accuracy": round(acc, 3),
            "feature_importance": importances,
        }

    return results
