# Build an ML Pipeline for Short-term Rental Prices in NYC

This repository contains an end-to-end machine learning pipeline for predicting short-term rental prices in New York City. The pipeline is orchestrated with MLflow and Hydra, and each step logs artifacts and metrics to Weights & Biases (W&B).

## Weights & Biases Project (Public)
https://wandb.ai/urwah-khatoon-/nyc_airbnb/overview

## GitHub Repository (v1.0.1 Release)
https://github.com/urwa95/Build-end-to-end-ML-pipeline

## Repository Structure
```
build-ml-pipeline-for-short-term-rental-prices/
├── src/
│   ├── basic_cleaning/
│   │   ├── MLproject
│   │   ├── conda.yml
│   │   └── run.py
│   ├── data_check/
│   │   ├── MLproject
│   │   ├── conda.yml
│   │   └── run.py
│   │   └── test_data.py
│   ├── train_val_test_split/  ← external component from W&B
│   ├── train_random_forest/
│   │   ├── MLproject
│   │   ├── conda.yml
│   │   └── run.py
│   ├── test_regression_model/
│   │   ├── MLproject
│   │   ├── conda.yml
│   │   └── run.py
│   └── eda/
│       └── eda.ipynb
├── main.py  ← orchestrates all steps via Hydra/MLflow
├── config.yaml  ← Hydra configuration file
└── README.md  ← you are here
```

## Live Project on W&B
A public W&B project has been set up so you can verify each step’s outputs and artifacts. You can view all runs and artifacts here:
**W&B Project URL:** https://wandb.ai/urwa95/nyc_airbnb

## GitHub Repository
All code is hosted publicly on GitHub. Feel free to clone or fork:
**GitHub Repo URL:** https://github.com/urwa95/Build-end-to-end-ML-pipeline

## Installation & Setup

1. Clone this repository
    ```bash
    git clone git@github.com:urwa95/Build-end-to-end-ML-pipeline.git
    cd Build-end-to-end-ML-pipeline
    ```

2. Create (or activate) your Python environment (using Conda or venv). Example with Conda:
    ```bash
    conda create -n nyc_airbnb python=3.9
    conda activate nyc_airbnb
    ```

3. Install MLflow, Hydra, and other dependencies. Each individual step has its own `conda.yml` or `requirements.txt`. You can install a global set of requirements by running:
    ```bash
    pip install mlflow hydra-core wandb pandas scikit-learn omegaconf
    ```

4. Authenticate W&B (if you haven’t already):
    ```bash
    wandb login
    ```
    This will prompt you for your W&B API key.

5. Configure your W&B project name. By default, `main.py` sets:
    ```
    WANDB_PROJECT=nyc_airbnb
    WANDB_RUN_GROUP=<experiment_name from config.yaml>
    ```

---

# Build an ML Pipeline for Short-Term Rental Prices in NYC

You are working for a property management company renting rooms and properties for short periods of time on various rental platforms. You need to estimate the typical price for a given property based on the price of similar properties. Your company receives new data in bulk every week. The model needs to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

---

## Table of contents
1. [Introduction](#introduction)
2. [Preliminary steps](#preliminary-steps)
3. [Instructions](#instructions)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Cleaning](#data-cleaning)
6. [Data Testing](#data-testing)
7. [Data Splitting](#data-splitting)
8. [Train Random Forest](#train-random-forest)
9. [Optimize Hyperparameters](#optimize-hyperparameters)
10. [Select the Best Model](#select-the-best-model)
11. [Test Set Verification](#test-set-verification)
12. [Visualize the Pipeline](#visualize-the-pipeline)
13. [Release the Pipeline](#release-the-pipeline)
14. [Train the Model on a New Data Sample](#train-the-model-on-a-new-data-sample)
15. [Acknowledgments](#acknowledgments)

---

## Introduction
The main pipeline is defined in `main.py`. It orchestrates the following steps via Hydra and MLflow:
- **Download data** (get_data component)
- **Basic Cleaning** (`src/basic_cleaning/run.py`)
- **Data Checking** (`src/data_check/run.py`)
- **Data Splitting** (train_val_test_split component)
- **Train Random Forest** (`src/train_random_forest/run.py`)
- **Test Regression Model** (`src/test_regression_model/run.py`)

All steps log artifacts and metrics to W&B and are composed as MLflow steps.

---

## Preliminary steps

### Fork the Starter Kit
Go to:  
`https://github.com/udacity/nd0821-c2-build-model-workflow-starter`  
Click **Fork** in the upper right. Clone your fork locally:
```bash
git clone https://github.com/[your_username]/nd0821-c2-build-model-workflow-starter.git
cd nd0821-c2-build-model-workflow-starter
```

Commit and push frequently with meaningful messages.

### Create environment
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Go to `https://wandb.ai/authorize` and copy your API key. Run:
```
wandb login [your API key]
```

### Cookie Cutter (Optional)
A cookie cutter template is provided to scaffold new MLflow steps:
```bash
cookiecutter cookie-mlflow-step -o src
step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and saves the results in Weights & Biases
parameters [parameter1,parameter2]: input_artifact,output_artifact,output_type,output_description,min_price,max_price
```

---

## Instructions

### Running the Pipeline
From the root directory:
```bash
mlflow run .
```
This runs the entire pipeline.  

To run specific steps:
```bash
mlflow run . -P steps=download,basic_cleaning
```
Override any config parameter via `hydra_options`. Example:
```bash
mlflow run .   -P steps=download,basic_cleaning   -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

### Pre-existing Components
External reusable components hosted on GitHub:
- **get_data** (download step)
- **train_val_test_split** (data splitting)

You can call:
```python
_ = mlflow.run(
    f"{config['main']['components_repository']}/get_data",
    "main",
    parameters={
        "sample": config["etl"]["sample"],
        "artifact_name": "sample.csv",
        "artifact_type": "raw_data",
        "artifact_description": "Raw file as downloaded"
    },
)
```
where `config['main']['components_repository']` points to the Udacity starter repo.

---

## Exploratory Data Analysis (EDA)

1. Run:
    ```bash
    mlflow run . -P steps=download
    ```
2. Run the EDA notebook:
    ```bash
    mlflow run src/eda
    ```
    - Jupyter opens. Fetch `sample.csv` with:
      ```python
      import wandb, pandas as pd
      run = wandb.init(project="nyc_airbnb", group="eda", save_code=True)
      local_path = wandb.use_artifact("sample.csv:latest").file()
      df = pd.read_csv(local_path)
      ```
    - Use `pandas_profiling` for a quick profile:
      ```python
      import pandas_profiling
      profile = pandas_profiling.ProfileReport(df)
      profile.to_widgets()
      ```
    - Drop outliers and convert date:
      ```python
      df = df[df['price'].between(10, 350)].copy()
      df['last_review'] = pd.to_datetime(df['last_review'])
      ```
    - Finish the run: `run.finish()`

---

## Data Cleaning

Transfer the cleaning in EDA to `src/basic_cleaning/run.py`.  
- Parameters: `input_artifact`, `output_artifact`, `output_type`, `output_description`, `min_price`, `max_price`.
- Download sample from W&B, drop outliers, drop out-of-NYC coordinates:
  ```python
  df = df[df['price'].between(args.min_price, args.max_price)]
  df = df[df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.50, 40.90)]
  ```
- Save to `clean_sample.csv`, log artifact:
  ```python
  df.to_csv("clean_sample.csv", index=False)
  artifact = wandb.Artifact(
      args.output_name, 
      type=args.output_type, 
      description=args.output_description
  )
  artifact.add_file("clean_sample.csv")
  run.log_artifact(artifact)
  ```

Update `src/basic_cleaning/MLproject` with parameter types.  
Add step to `main.py`:
```python
if "basic_cleaning" in active_steps:
    _ = mlflow.run(
        os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
        "main",
        parameters={
            "input_artifact": "sample.csv:latest",
            "output_name": "clean_sample.csv",
            "output_type": "clean_sample",
            "output_description": "Data with outliers and null values removed",
            "min_price": config['etl']['min_price'],
            "max_price": config['etl']['max_price']
        },
    )
```

Run:
```bash
mlflow run . -P steps=basic_cleaning
```
Verify `clean_sample.csv` appears in W&B.

---

## Data Testing
Tag the latest `clean_sample.csv` as `reference` in W&B (Artifacts → clean_sample → +Aliases → `reference`).  
Implement tests in `src/data_check/test_data.py`:
```python
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data, min_price, max_price):
    assert data['price'].between(min_price, max_price).all()
```
Add `data_check` to `main.py` with parameters:
- `csv`: `clean_sample.csv:latest`
- `ref`: `clean_sample.csv:reference`
- `kl_threshold`: `config["data_check"]["kl_threshold"]`
- `min_price`, `max_price` from config.

Run:
```bash
mlflow run . -P steps=data_check
```
All tests should pass.

---

## Data Splitting
Use the `train_val_test_split` component:
```python
if "data_split" in active_steps:
    _ = mlflow.run(
        f"{config['main']['components_repository']}/train_val_test_split",
        "main",
        parameters={
            "input": "clean_sample.csv:latest",
            "test_size": config["modeling"]["test_size"],
            "random_seed": config["modeling"]["random_seed"],
            "stratify_by": config["modeling"]["stratify_by"],
        },
    )
```
Run:
```bash
mlflow run . -P steps=data_split
```
You should see W&B artifacts: `trainval_data.csv` and `test_data.csv`.

---

## Train Random Forest
Complete `src/train_random_forest/run.py`. Key steps:
1. Download `trainval_data.csv` from W&B.
2. Implement `get_inference_pipeline`:
   ```python
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestRegressor

   def get_inference_pipeline(rf_config):
       non_ordinal_categorical_preproc = Pipeline([
           ("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))
       ])
       sk_pipe = Pipeline([
           ("preprocessing", non_ordinal_categorical_preproc),
           ("model", RandomForestRegressor(
               n_estimators=rf_config["n_estimators"],
               max_features=rf_config["max_features"],
               random_state=rf_config["random_state"]
           ))
       ])
       return sk_pipe
   ```
3. In `go`, fit pipeline, log MAE to W&B, export with:
   ```python
   mlflow.sklearn.save_model(sk_pipe, "model_export")
   artifact = wandb.Artifact(args.output_artifact, type="model_export")
   artifact.add_dir("model_export")
   run.log_artifact(artifact)
   ```
Update `src/train_random_forest/MLproject` accordingly.  
Add to `main.py`:
```python
if "train_random_forest" in active_steps:
    _ = mlflow.run(
        os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
        "main",
        parameters={
            "trainval_data": "trainval_data.csv:latest",
            "rf_config": rf_config,
            "output_artifact": "random_forest_export"
        },
    )
```
Run:
```bash
mlflow run . -P steps=train_random_forest
```
A W&B artifact `random_forest_export` should appear.

---

## Optimize Hyperparameters
Use Hydra multi-run:
```bash
mlflow run .   -P steps=train_random_forest   -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```
Check W&B table for multiple runs and MAE comparisons.

---

## Select the Best Model
In W&B, switch to Table view. Show columns `ID`, `Job Type`, `max_depth`, `n_estimators`, `mae`, `r2`. Sort `mae` ascending.  
Click best run → Artifacts → select `model_export` → add tag `prod`.

---

## Test Set Verification
Implement `test_regression_model` step in `main.py`. Use W&B component:
```python
if "test_regression_model" in active_steps:
    _ = mlflow.run(
        os.path.join(hydra.utils.get_original_cwd(), "src", "test_regression_model"),
        "main",
        parameters={
            "mlflow_model": "random_forest_export:prod",
            "test_artifact": "test_data.csv:latest",
            "kl_threshold": config["data_check"]["kl_threshold"],
            "min_price": config["etl"]["min_price"],
            "max_price": config["etl"]["max_price"]
        },
    )
```
Run:
```bash
mlflow run . -P steps=test_regression_model
```
Ensure no overfitting: test MAE ≈ validation MAE.

---

## Visualize the Pipeline
On W&B, go to **Artifacts** → select `model_export` → **Graph view**. You’ll see the pipeline DAG.

---

## Release the Pipeline
1. Ensure everything works on `sample.csv`:
   ```bash
   mlflow run . -P steps="download,basic_cleaning,data_check,data_split,train_random_forest"
   ```
2. Tag `v1.0.0`:
   ```bash
   git checkout main
   git pull origin main
   git tag -a v1.0.0 -m "Release v1.0.0: pipeline works on sample.csv"
   git push origin v1.0.0
   ```
3. On GitHub, verify v1.0.0 and publish a release.

---

## Train the Model on a New Data Sample
1. Place `sample2.csv` in your project root or in `components/get_data/data/`.
2. Run release-locked pipeline (v1.0.0) – expect failure at data tests:
   ```bash
   git checkout v1.0.0
   mlflow run .      -P steps="download,basic_cleaning,data_check,data_split,train_random_forest"      -P etl.sample="sample2.csv"
   ```
   The `test_proper_boundaries` check should fail because of out-of-NYC points.
3. Add NYC bounds filter to `src/basic_cleaning/run.py` (as shown above).
4. Commit changes, bump to `v1.0.1`:
   ```bash
   git checkout main
   git pull origin main
   git add src/basic_cleaning/run.py
   git commit -m "Release v1.0.1: add NYC-bounds filter"
   git tag -a v1.0.1 -m "Release v1.0.1: add NYC-bounds filter"
   git push origin main --tags
   ```
5. Now `v1.0.1` runs successfully on both `sample.csv` and `sample2.csv`:
   ```bash
   git checkout v1.0.1
   mlflow run . -P steps="download,basic_cleaning,data_check,data_split,train_random_forest" -P etl.sample="sample2.csv"
   ```

---

## What You Should See on W&B
- **Artifacts**:
  - `clean_sample.csv` (type `clean_sample`), tags: `latest`, `reference`
  - `trainval_data.csv`, tag: `latest`
  - `test_data.csv`, tag: `latest`
  - `random_forest_export` (type `model_export`), tags: `latest`, `prod`
  - `test_regression_results` (type `evaluation`) – MAE and R² on the hold-out test set (optional)

- **Runs**:
  - Separate W&B runs for each MLflow step, with logged metrics and parameters.

---

## Configuration (`config.yaml`)
All parameters (e.g. `min_price`, `max_price`, RF hyperparameters) are defined here. Override via Hydra:
```bash
mlflow run .   -P steps=train_random_forest   -P hydra_options="modeling.random_forest.max_features=0.5 modeling.random_forest.n_estimators=200 -m"
```
```yaml
main:
  project_name: "nyc_airbnb"
  experiment_name: "basic_run"
  components_repository: "https://github.com/urwa95/Build-end-to-end-ML-pipeline#components"
  steps: "all"  # or comma-separated list

etl:
  sample: "sample.csv"
  min_price: 10
  max_price: 350

data_check:
  kl_threshold: 0.2

modeling:
  test_size: 0.2
  random_seed: 42
  stratify_by: "neighbourhood_group"

random_forest:
  n_estimators: 100
  max_features: 0.33
  random_state: 42
```

---

## Sample Commands

- **Run all steps**:
  ```bash
  mlflow run . -P steps="download,basic_cleaning,data_check,data_split,train_random_forest"
  ```
- **Run only data check**:
  ```bash
  mlflow run . -P steps="data_check"
  ```
- **Train Random Forest only**:
  ```bash
  mlflow run . -P steps="train_random_forest"
  ```
- **Test production model**:
  ```bash
  mlflow run . -P steps="test_regression_model"
  ```

---

## Acknowledgments
This pipeline was built as part of a Data Engineering Nanodegree, leveraging:
- **MLflow** for experiment tracking
- **Hydra** for configuration and hyperparameter tuning
- **Weights & Biases (W&B)** for artifact and metric logging
- **Scikit-learn** for modeling

© 2025 Urwa Khatoon
