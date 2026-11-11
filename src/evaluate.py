
import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "sentiment labelled sentences"

DATA_FILES = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt"
]

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
    data = data.dropna()
    data = data[data["text"] != ""]

    return data

def main():
    data = load_data()

    X = data["text"]
    y = data["label"]

    # Повторяем разбиение, использованное при обучении.
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

    # Загружаем ранее экспортированную модель.
    model = mlflow.sklearn.load_model(
        str(PROJECT_DIR / "exported_model")
    )

    predictions = model.predict(X_test)

    # Расчет итоговых метрик.
    metrics = {
        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "test_precision_macro": float(
            precision_score(
                y_test, predictions, average="macro"
            )
        ),
        "test_recall_macro": float(
            recall_score(
                y_test, predictions, average="macro"
            )
        ),
        "test_f1_macro": float(
            f1_score(
                y_test, predictions, average="macro"
            )
        ),
        "test_size": len(X_test)
    }

    metrics_path = PROJECT_DIR / "test_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    # Построение матрицы ошибок.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["negative", "positive"]
    )

    display.plot(cmap="Blues")
    plt.title("Test confusion matrix")
    plt.tight_layout()

    matrix_path = PROJECT_DIR / "confusion_matrix_test.png"
    plt.savefig(matrix_path)
    plt.close()

    print("Test metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["negative", "positive"]
        )
    )

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {matrix_path}")

if __name__ == "__main__":
    main()