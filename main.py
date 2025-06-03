import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include test_regression_model in the default steps so it won’t run accidentally.
    "test_regression_model",
]

@hydra.main(config_name="config")
def go(config: DictConfig):
    """Orchestrate the entire pipeline based on the selected steps."""
    # Retrieve which steps to run from the Hydra overrides; if none specified, default to all except test_regression_model
    active_steps = config["main"]["steps"]
    if not active_steps:
        active_steps = [s for s in _steps if s != "test_regression_model"]

    # 1) DOWNLOAD step
    if "download" in active_steps:
        # This component lives in components/get_data
        _ = mlflow.run(
            uri=f"{config['main']['components_repository']}/get_data",
            entry_point="main",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw listings data as downloaded from source",
            },
        )

    # 2) BASIC_CLEANING step
    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            uri=os.path.join(
                hydra.utils.get_original_cwd(), "src", "basic_cleaning"
            ),
            entry_point="main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_name": config["etl"]["cleaned_name"],
                "output_type": config["etl"]["cleaned_type"],
                "output_description": config["etl"]["cleaned_description"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # 3) DATA_CHECK step
    if "data_check" in active_steps:
        _ = mlflow.run(
            uri=os.path.join(
                hydra.utils.get_original_cwd(), "src", "data_check"
            ),
            entry_point="main",
            parameters={
                "csv": f"{config['etl']['cleaned_name']}:latest",
                "ref": f"{config['etl']['cleaned_name']}:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # 4) DATA_SPLIT step
    if "data_split" in active_steps:
        _ = mlflow.run(
            uri=f"{config['main']['components_repository']}/train_val_test_split",
            entry_point="main",
            parameters={
                "input": f"{config['etl']['cleaned_name']}:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
        )

    # 5) TRAIN_RANDOM_FOREST step
    if "train_random_forest" in active_steps:
        _ = mlflow.run(
            uri=os.path.join(
                hydra.utils.get_original_cwd(), "src", "train_random_forest"
            ),
            entry_point="main",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "rf_config": json.dumps(config["modeling"]["random_forest"]),
                "output_artifact": "random_forest_export",
            },
        )

    # 6) TEST_REGRESSION_MODEL step
    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            uri=os.path.join(
                hydra.utils.get_original_cwd(), "src", "test_regression_model"
            ),
            entry_point="main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_artifact": "test_data.csv:latest",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )


if __name__ == "__main__":
    go()
