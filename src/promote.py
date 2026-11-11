
import argparse
import os

import mlflow
from mlflow import MlflowClient


parser = argparse.ArgumentParser()

parser.add_argument("--version", required=True)
parser.add_argument("--alias", default="champion")

args = parser.parse_args()

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

client = MlflowClient()

model_name = "sentiment-classifier"

version = client.get_model_version(
    model_name,
    args.version
)

client.set_registered_model_alias(
    model_name,
    args.alias,
    args.version
)

print(
    f"{args.alias} -> version {version.version}, "
    f"run {version.run_id}"
)
