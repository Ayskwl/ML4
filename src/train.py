
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "sentiment labelled sentences"

DATA_FILES = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt"
]

RANDOM_STATE = 42
EXPERIMENT_NAME = "sentiment-classification"


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

    data = data.dropna()
    data = data[data["text"] != ""]

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("Обнаружены недопустимые метки классов.")

    return data


def split_data(data):
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_data_hash():
    sha256 = hashlib.sha256()

    for filename in DATA_FILES:
        file_path = DATA_DIR / filename

        with open(file_path, "rb") as file:
            sha256.update(file.read())

    return sha256.hexdigest()


def create_model(algorithm, c, ngram_min, ngram_max):
    if algorithm == "lr":
        classifier = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    elif algorithm == "nb":
        classifier = MultinomialNB()
    else:
        raise ValueError("Неизвестный алгоритм.")

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(ngram_min, ngram_max)
            )
        ),
        ("classifier", classifier)
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        choices=["lr", "nb"],
        required=True
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=1)

    parser.add_argument(
        "--register",
        action="store_true",
        help="Зарегистрировать обученную модель в MLflow Model Registry"
    )

    args = parser.parse_args()

    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("Некорректное значение ngram_range.")

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(data)

    model = create_model(
        args.algorithm,
        args.c,
        args.ngram_min,
        args.ngram_max
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        f1_macro = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("C", args.c if args.algorithm == "lr" else "not_applicable")
        mlflow.log_param(
            "ngram_range",
            f"({args.ngram_min}, {args.ngram_max})"
        )
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("validation_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_f1_macro", f1_macro)

        mlflow.set_tag("dataset", "Sentiment Labelled Sentences")
        mlflow.set_tag("variant", "1")

        metadata = {
            "dataset": "Sentiment Labelled Sentences",
            "records": len(data),
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test),
            "random_state": RANDOM_STATE,
            "data_sha256": calculate_data_hash()
        }

        metadata_path = PROJECT_DIR / "split.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

        mlflow.log_artifact(str(metadata_path))

        matrix = confusion_matrix(y_val, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["negative", "positive"]
        )

        display.plot(cmap="Blues")
        plt.title("Validation confusion matrix")
        plt.tight_layout()

        matrix_path = PROJECT_DIR / "confusion_matrix.png"
        plt.savefig(matrix_path)
        plt.close()

        mlflow.log_artifact(str(matrix_path))

        input_example = ["This product is excellent."]
        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature
        )

        if args.register:
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="sentiment-classifier"
            )
            print(f"REGISTERED_VERSION={registered_model.version}")

        print(f"Run ID: {run.info.run_id}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro F1: {f1_macro:.4f}")
        print(f"Model URI: runs:/{run.info.run_id}/model")


if __name__ == "__main__":
    main()
