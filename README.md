Build an ML Pipeline for Short-term Rental Prices in NYC

This repository contains an end-to-end machine learning pipeline for predicting short-term rental prices in New York City. The pipeline is orchestrated with MLflow and Hydra, and each step logs artifacts and metrics to Weights & Biases (W&B).

📂 Repository Structure

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
│   ├── train_val_test_split/        ← external component from W&B
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
├── main.py                          ← orchestrates all steps via Hydra/MLflow
├── config.yaml                      ← Hydra configuration file
└── README.md                        ← you are here

🚀 Live Project on W&B

A public W&B project has been set up so you can verify each step’s outputs and artifacts. You can view all runs and artifacts here:

W&B Project URL:https://wandb.ai/urwa95/nyc_airbnb

🔗 GitHub Repository

All code is hosted publicly on GitHub. Feel free to clone or fork:

GitHub Repo URL:https://github.com/urwa95/Build-end-to-end-ML-pipeline

🔧 Installation & Setup

Clone this repository

git clone git@github.com:urwa95/Build-end-to-end-ML-pipeline.git
cd Build-end-to-end-ML-pipeline

Create (or activate) your Python environment (using Conda or venv).Example with Conda:

conda create -n nyc_airbnb python=3.9
conda activate nyc_airbnb

Install MLflow and Hydra (and other dependencies).Each individual step has its own conda.yml or requirements.txt; you can install a global set of requirements by running:

pip install mlflow hydra-core wandb pandas scikit-learn omegaconf

Authenticate W&B (if you haven’t already):

wandb login

This will prompt you for your W&B API key.

Configure your W&B project nameBy default, main.py sets:

WANDB_PROJECT=nyc_airbnb
WANDB_RUN_GROUP=<experiment_name from config.yaml>

You can override these in config.yaml.

⚙️ How to Run the Pipeline

All steps are orchestrated through main.py via MLflow. You can run individual steps or the entire pipeline. Below are the most common commands:

Run all steps in sequence (download → clean → check → split → train):

mlflow run . -P steps="download,basic_cleaning,data_check,data_split,train_random_forest"

This will:

Download a sample of the Airbnb data

Perform basic cleaning

Run data validation tests

Split into train/validation and test sets

Train a Random Forest and log model artifacts to W&B

Run only the data-checking step:

mlflow run . -P steps="data_check"

This uses the latest clean_sample.csv:latest artifact and the clean_sample.csv:reference tag to run distribution tests.

Train a Random Forest only (assuming data is already split):

mlflow run . -P steps="train_random_forest"

Test the “production” model against the test set (after you’ve manually tagged your best model as :prod in W&B):

mlflow run . -P steps="test_regression_model"

This step loads random_forest_export:prod and evaluates it on test_data.csv:latest. You must first tag a model version as prod in the W&B UI.

📝 Step-by-Step Pipeline Overview

Download

Component: get_data (external MLflow step from our components repo)

Outputs: sample.csv (uploaded to W&B as raw_data)

Basic Cleaning

Location: src/basic_cleaning/run.py

Reads: sample.csv:latest

Parameters: min_price, max_price (from config.yaml)

Outputs: clean_sample.csv as a W&B artifact

Data Checking

Location: src/data_check/run.py

Reads: clean_sample.csv:latest and clean_sample.csv:reference

Performs KL divergence and range tests (test_row_count, test_price_range)

Fails if checks do not pass

Data Splitting

Component: train_val_test_split (external)

Reads: clean_sample.csv:latest

Parameters: test_size, random_seed, stratify_by (from config.yaml)

Outputs: trainval_data.csv and test_data.csv (uploaded to W&B)

Train Random Forest

Location: src/train_random_forest/run.py

Reads: trainval_data.csv:latest and test_data.csv:latest

Builds a pipeline with imputation/preprocessing → RandomForestRegressor

Logs MAE and R² to W&B

Exports the trained model via mlflow.sklearn.save_model(...)

Outputs: A random_forest_export artifact (MLflow “model_export”)

Test Regression Model

Location: src/test_regression_model/run.py

Reads: random_forest_export:prod (model artifact) and test_data.csv:latest

Evaluates on the test set (computes MAE and R²)

Logs metrics to W&B

📄 Configuration (config.yaml)

Below is a minimal excerpt of config.yaml. All parameters (e.g. min_price, max_price, hyperparameters for the RF, paths to components, etc.) are defined here. You can override any of these on the command line via Hydra, for example:

mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_features=0.5 modeling.random_forest.n_estimators=200 -m"

main:
  project_name: "nyc_airbnb"
  experiment_name: "basic_run"
  components_repository: "https://github.com/urwa95/your-ml-components-repo"  # replace with actual
  steps: "all"                         # or comma-separated list

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

📌 What You Should See on W&B

By the time all steps have run, you should see the following artifacts uploaded and tagged in your W&B project (nyc_airbnb):

clean_sample.csv (type clean_sample)

Tags: latest, reference

trainval_data.csv (type dataset)

Tag: latest

test_data.csv (type dataset)

Tag: latest

random_forest_export (type model_export)

Tags: latest, and once you pick the best run, prod

test_regression_results (type evaluation)

Contains MAE/R² on the hold-out test set (optional, if that step logs an artifact)

Additionally, you will see individual W&B runs for each MLflow step (e.g. one run for basic_cleaning, one for data_check, one for train_random_forest, etc.), each with its own table of metrics and parameters.

📦 Cutting a GitHub Release

Make sure everything is working on sample.csvRun:

mlflow run . -P steps="download,basic_cleaning,data_check,data_split,train_random_forest"

Confirm that all W&B artifacts appear and no step errors out.

Tag the current commit as v1.0.0

git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release v1.0.0: pipeline works on sample.csv"
git push origin v1.0.0

This creates a GitHub “Release” (you can also create the release via the GitHub UI if you prefer).

Publish the tag on GitHubGo to your repository’s “Releases” page on GitHub, verify that v1.0.0 is there, and draft/publish a new release if desired.

After this point, anyone can clone or check out v1.0.0 and run the exact pipeline that “works on sample.csv.”

✏️ Next Steps (Train on a New Sample)

Fetch sample2.csv manually and place it in your project root or point the pipeline to it (for example, put it in /data/sample2.csv).

Run the release-locked pipeline against sample2.csv (this should fail initially because the cleaning does not yet handle out-of-NYC locations):

git checkout v1.0.0
mlflow run . \
  -P steps="download,basic_cleaning,data_check,data_split,train_random_forest" \
  -P etl.sample="sample2.csv"


3. **Add the extra boundary check for NYC** in `src/basic_cleaning/run.py` (e.g. drop listings with lat/lon outside NYC).

4. **Bump to v1.0.1**  
   ```bash
   git checkout main
   # implement the new cleaning logic
   git add src/basic_cleaning/run.py
   git commit -m "Handle out-of-NYC listings in basic_cleaning"
   git tag -a v1.0.1 -m "Release v1.0.1: support sample2.csv"
   git push origin main --tags

Now v1.0.1 should run successfully on both sample.csv and sample2.csv.

🙏 Acknowledgments

This pipeline was built as part of the Data Engineering Nanodegree (or similar), adapted to predict Airbnb rental prices. It leverages:

MLflow for experiment tracking and reproducible runs

Hydra for flexible configuration and hyperparameter tuning

Weights & Biases (W&B) for logging artifacts and metrics

Scikit-learn for modeling

Feel free to open issues or pull requests if you see improvements!

© 2025 Urwa Khatoon

