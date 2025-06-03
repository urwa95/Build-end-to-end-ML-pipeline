import argparse
import json
import pandas as pd
import wandb
import mlflow
import shutil
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

def get_feature_pipeline(rf_config, df):
    """
    1) Identify numerical vs. categorical features
    2) Numeric columns: impute missing values with median
    3) Categorical columns: impute with most frequent, then one-hot
    4) Combine into ColumnTransformer
    5) Add RandomForestRegressor
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "price"]  # exclude target

    numeric_preproc = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_preproc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preproc, num_cols),
            ("cat", categorical_preproc, cat_cols),
        ],
        remainder="drop"
    )

    with open(rf_config) as f:
        rf_params = json.load(f)
    rf = RandomForestRegressor(**rf_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", rf),
        ]
    )
    return pipeline

def go(args):
    # Clean up any leftover model_export folder
    if os.path.isdir("model_export"):
        shutil.rmtree("model_export")

    run = wandb.init(job_type="train_random_forest")
    run.config.update({
        "trainval_artifact": args.trainval_artifact,
        "test_artifact": args.test_artifact,
        "rf_config": args.rf_config
    })

    # 1) Download train+val CSV
    artifact_trainval = run.use_artifact(args.trainval_artifact)
    trainval_local = artifact_trainval.file()
    df_trainval = pd.read_csv(trainval_local)

    # 2) Split into X/y
    X_trainval = df_trainval.drop(columns=["price"])
    y_trainval = df_trainval["price"]

    # 3) Build pipeline
    pipeline = get_feature_pipeline(args.rf_config, df_trainval)

    # 4) Fit
    pipeline.fit(X_trainval, y_trainval)

    # 5) Download test CSV
    artifact_test = run.use_artifact(args.test_artifact)
    test_local = artifact_test.file()
    df_test = pd.read_csv(test_local)
    X_test = df_test.drop(columns=["price"])
    y_test = df_test["price"]

    # 6) Predict & log metrics (use run.log instead of run.log_metric)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    run.log({ "mae": mae, "r2": r2 })

    # 7) Save model with MLflow (fresh directory ensured above)
    model_uri = "model_export"
    mlflow.sklearn.save_model(pipeline, model_uri)

    # 8) Log model artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random forest regression model",
    )
    artifact.add_dir(model_uri)
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest")
    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="W&B artifact for train+val data (e.g., trainval_data.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--test_artifact",
        type=str,
        help="W&B artifact for test data (e.g., test_data.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--rf_config",
        type=str,
        help="Path to JSON file with RF params",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output model artifact (e.g., random_forest_export)",
        required=True,
    )
    args = parser.parse_args()
    go(args)
