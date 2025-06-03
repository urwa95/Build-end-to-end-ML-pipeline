# src/test_regression_model/run.py

import argparse
import pandas as pd
import wandb
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score

def go(args):
    # 1) Start a W&B run for this test step
    run = wandb.init(job_type="test_regression_model")
    run.config.update({
        "mlflow_model": args.mlflow_model,
        "test_artifact": args.test_artifact,
        "kl_threshold": args.kl_threshold,
        "min_price": args.min_price,
        "max_price": args.max_price,
    })

    # 2) Download & load the test dataset from W&B
    test_art = run.use_artifact(args.test_artifact)   # e.g. "test_data.csv:latest"
    test_path = test_art.file()                       # local path, e.g. /tmp/.../test_data.csv
    df_test = pd.read_csv(test_path)

    # 3) Filter out any rows whose "price" is outside [min_price, max_price]
    df_test = df_test[
        (df_test.price >= args.min_price) &
        (df_test.price <= args.max_price)
    ]

    # 4) Separate features/target
    X_test = df_test.drop(columns=["price"])
    y_test = df_test["price"]

    # 5) Download & load the production‐tagged model artifact
    #    After download(), model_dir points to something like
    #      /tmp/.../random_forest_export:v13
    model_art = run.use_artifact(args.mlflow_model)   # e.g. "random_forest_export:prod"
    model_dir = model_art.download()                  # folder containing MLmodel, model.pkl, etc.

    # 6) Load the MLmodel directly from that local folder (no "file://")
    model = mlflow.pyfunc.load_model(model_dir)

    # 7) Run inference
    preds = model.predict(X_test)

    # 8) Compute test‐set metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # 9) Log those metrics to W&B
    run.log({"mae": mae, "r2": r2})

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a production‐tagged regression model on the test set"
    )
    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="W&B artifact for the production model (e.g. random_forest_export:prod)",
        required=True,
    )
    parser.add_argument(
        "--test_artifact",
        type=str,
        help="W&B artifact for the test CSV (e.g. test_data.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        help="KL divergence threshold for distribution checks (not used here)",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price to keep in the test set",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price to keep in the test set",
        required=True,
    )
    args = parser.parse_args()
    go(args)

