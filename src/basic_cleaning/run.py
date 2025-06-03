import argparse
import pandas as pd
import wandb


def go(args):
    # Start a new W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(
        {
            "input_artifact": args.input_artifact,
            "output_artifact": args.output_artifact,
            "min_price": args.min_price,
            "max_price": args.max_price,
        }
    )

    # Download the input artifact (raw data)
    artifact = run.use_artifact(args.input_artifact)
    local_path = artifact.file()
    df = pd.read_csv(local_path)

    # Basic cleaning:
    #   1) Drop rows with price outside [min_price, max_price]
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    #   2) Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save cleaned data locally
    df.to_csv("clean_sample.csv", index=False)

    # Create and log a new W&B artifact for the cleaned data
    cleaned_artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    cleaned_artifact.add_file("clean_sample.csv")
    run.log_artifact(cleaned_artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning step")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="W&B artifact name for raw data (e.g., sample.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for cleaned data artifact (e.g., clean_sample.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for cleaned data artifact (e.g., clean_sample)",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned data artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to keep",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to keep",
        required=True,
    )
    args = parser.parse_args()
    go(args)

