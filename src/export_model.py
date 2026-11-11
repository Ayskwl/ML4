
import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "sentiment labelled sentences"

DATA_FILES = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt"
]

MODEL_NAME = "sentiment-classifier"
ALIAS = "champion"
RANDOM_STATE = 42


def load_data():
    frames = []

    for filename in DATA_FILES:
        file_path = DATA_DIR / filename

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["text", "label"],
            encoding="utf-8",
            quoting=3
        )

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(int)

    return data


def main():
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )

    client = MlflowClient()

    version_info = client.get_model_version_by_alias(
        MODEL_NAME,
        ALIAS
    )

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{version_info.version}"
    )

    data = load_data()

    X = data["text"]
    y = data["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    output_dir = PROJECT_DIR / "exported_model"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    mlflow.sklearn.save_model(
        model,
        str(output_dir),
        pip_requirements=(
            PROJECT_DIR / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
    )

    run = client.get_run(version_info.run_id)
    params = run.data.params

    metadata = {
        "name": MODEL_NAME,
        "version": version_info.version,
        "alias": ALIAS,
        "run_id": version_info.run_id,
        "algorithm": params.get("algorithm"),
        "C": params.get("C"),
        "ngram_range": params.get("ngram_range"),
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
        "classes": {
            "0": "negative",
            "1": "positive"
        }
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    sample_texts = X_test.iloc[:3].tolist()

    request = {
        "texts": sample_texts
    }

    (PROJECT_DIR / "sample_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    expected_predictions = model.predict(sample_texts).tolist()

    (PROJECT_DIR / "expected_predictions.json").write_text(
        json.dumps(expected_predictions, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=4))
    print("Model exported successfully.")


if __name__ == "__main__":
    main()
