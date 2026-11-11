
import json
import logging
import time
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    BASE_DIR / "exported_model"
    if (BASE_DIR / "exported_model").exists()
    else BASE_DIR.parent / "exported_model"
)

# Загрузка модели и метаданных при запуске приложения.
model = mlflow.sklearn.load_model(str(MODEL_DIR))
metadata = json.loads(
    (MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
)

app = FastAPI(
    title="Sentiment Classification API",
    version="1.0"
)

log = logging.getLogger("uvicorn.error")

class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts):
        for text in texts:
            if not text.strip():
                raise ValueError("Texts must not be empty")
            if len(text) > 5000:
                raise ValueError("Text must not exceed 5000 characters")
        return texts

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": metadata["version"]
    }

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict")
def predict(body: TextRequest):
    start = time.perf_counter()

    predictions = model.predict(body.texts).astype(int).tolist()

    class_names = [
        metadata["classes"][str(prediction)]
        for prediction in predictions
    ]

    elapsed = (time.perf_counter() - start) * 1000

    log.info(
        "prediction model_version=%s texts=%s latency_ms=%.3f",
        metadata["version"],
        len(body.texts),
        elapsed
    )

    return {
        "predictions": predictions,
        "class_names": class_names,
        "model_version": metadata["version"],
        "latency_ms": round(elapsed, 3)
    }