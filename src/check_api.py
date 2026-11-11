import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

request_data = json.loads(
    (BASE_DIR / "sample_request.json").read_text(encoding="utf-8")
)

expected = json.loads(
    (BASE_DIR / "expected_predictions.json").read_text(encoding="utf-8")
)

request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps(request_data).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))

actual = result["predictions"]

print("Expected predictions:", expected)
print("API predictions:", actual)
print("Model version:", result["model_version"])

if actual == expected:
    print("SUCCESS: predictions match")
else:
    print("ERROR: predictions do not match")
    raise SystemExit(1)
