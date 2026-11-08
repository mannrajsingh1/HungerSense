"""FastAPI service for accepting and storing hunger-related sensor readings.

The root endpoint reports that the API is running. The readings endpoint
validates sensor data, then adds each reading
to ``data/sensor_readings.csv``.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path
import csv

app = FastAPI(title="Hunger Detection API")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "sensor_readings.csv"


class SensorReading(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    participant_id: str = "P01"

    temperature_c: float | None = None
    heart_rate_bpm: float | None = None
    hrv_ms: float | None = None
    gsr: float | None = None

    sound_level: float | None = None
    air_quality: float | None = None
    pressure_hpa: float | None = None

    hunger: int | None = Field(default=None, ge=1, le=10)
    fullness: int | None = Field(default=None, ge=1, le=10)


@app.get("/")
def home():
    return {"message": "Hunger Detection API is running!"}


@app.post("/readings")
def add_reading(reading: SensorReading):
    record = reading.model_dump(mode="json")

    fieldnames = list(SensorReading.model_fields.keys())

    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)

    return {
        "success": True,
        "message": "Sensor reading saved",
        "reading": record
    }
