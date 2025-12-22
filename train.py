import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="Autonomous MLOps Platform")

MODEL_NAME = "StudentPerformanceModel"
EXPERIMENT_NAME = "student_performance_prediction"

# -------------------------------------------------
# Request Schemas
# -------------------------------------------------
class TrainRequest(BaseModel):
    max_retries: int = 3
    quality_threshold: float = 6.0  # MAE threshold

class PromoteRequest(BaseModel):
    version: int

# -------------------------------------------------
# Dataset Generator (Synthetic but realistic)
# -------------------------------------------------
def generate_dataset(n=300, seed=42):
    np.random.seed(seed)

    study_hours = np.random.randint(1, 10, n)
    attendance = np.random.randint(50, 100, n)
    prev_score = np.random.randint(40, 95, n)

    final_score = (
        2.5 * study_hours
        + 0.4 * attendance
        + 0.6 * prev_score
        + np.random.normal(0, 5, n)
    )

    final_score = np.clip(final_score, 0, 100)

    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "prev_score": prev_score,
        "final_score": final_score
    })

# -------------------------------------------------
# TRAIN + AUTO-RETRY
# -------------------------------------------------
@app.post("/train")
def train_model(request: TrainRequest):

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    best_mae = float("inf")
    best_version = None

    for attempt in range(1, request.max_retries + 1):

        dataset_size = 300 * attempt
        seed = 40 + attempt

        df = generate_dataset(dataset_size, seed)
        X = df[["study_hours", "attendance", "prev_score"]]
        y = df["final_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        with mlflow.start_run():
            mlflow.log_param("attempt", attempt)
            mlflow.log_param("dataset_size", dataset_size)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )

        # 🔥 GET ACTUAL MLflow VERSION (IMPORTANT FIX)
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
        current_version = max(int(v.version) for v in versions)

        if mae < best_mae:
            best_mae = mae
            best_version = current_version

        if mae <= request.quality_threshold:
            return {
                "status": "accepted",
                "mae": round(mae, 2),
                "recommended_version": current_version
            }

    return {
        "status": "rejected",
        "best_mae": round(best_mae, 2),
        "recommended_version": best_version
    }

# -------------------------------------------------
# PROMOTE WITH BACKUP
# -------------------------------------------------
@app.post("/promote")
def promote_model(req: PromoteRequest):

    client = MlflowClient()

    # Backup current production
    try:
        current_prod = client.get_model_version_by_alias(
            MODEL_NAME,
            "production"
        )
        client.set_registered_model_alias(
            MODEL_NAME,
            "previous_production",
            current_prod.version
        )
    except:
        pass  # first deployment

    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        str(req.version)
    )

    return {
        "status": "promoted",
        "production_version": req.version
    }

# -------------------------------------------------
# ROLLBACK
# -------------------------------------------------
@app.post("/rollback")
def rollback_model():

    client = MlflowClient()

    prev = client.get_model_version_by_alias(
        MODEL_NAME,
        "previous_production"
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        prev.version
    )

    return {
        "status": "rolled_back",
        "active_version": prev.version
    }
