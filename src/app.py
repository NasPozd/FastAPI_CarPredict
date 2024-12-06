"""
Модуль для предсказания цен автомобилей с использованием FastAPI и модели машинного обучения.
"""

import os
from io import StringIO
from typing import List

import joblib
import pandas as pd
import uvicorn
from transformer import CombinedTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class Item(BaseModel):
    """Определение модели Item (представляет данные об автомобиле)"""

    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class ItemCollection(BaseModel):
    """Определение коллекции Item (коллекция автомобилей)"""

    items: List[Item]


class PredictionService:
    """Сервис предсказаний"""

    def __init__(self, model_path: str):
        """Загрузка модели"""
        self.pipeline = joblib.load(model_path)

    def predict_single(self, item: Item) -> float:
        """Предсказание цены для одного автомобиля"""
        item_data = item.dict()
        item_data.pop("selling_price", None)
        data = pd.DataFrame([item_data])
        prediction = self.pipeline.predict(data)
        return float(prediction[0])

    def predict_multiple(self, items: List[Item]) -> List[float]:
        """Предсказание цен для нескольких автомобилей"""
        data = pd.DataFrame([item.dict() for item in items]).drop(
            columns=["selling_price"]
        )
        predictions = self.pipeline.predict(data)
        return predictions.tolist()


app = FastAPI()
model_path = os.getenv("MODEL_PATH", "C:/Users/naspo/VScode/FastAPI_CarPredict/model/model.pkl")
prediction_service = PredictionService(model_path=model_path)


@app.post("/predict_item", response_model=float)
def predict_item(item: Item) -> float:
    """Эндпоинт для предсказания цены одного автомобиля"""
    try:
        return prediction_service.predict_single(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/predict_items")
def predict_items(item_collection: ItemCollection) -> List[float]:
    """Эндпоинт для предсказания цен нескольких автомобилей"""
    try:
        items = item_collection.items
        predictions = prediction_service.predict_multiple(items)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/predict_csv")
async def upload_file(file: UploadFile = File(...)) -> FileResponse:
    """Эндпоинт для загрузки CSV файла и предсказания цен"""
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8"))).dropna()

        items = [Item(**row) for _, row in df.iterrows()]

        predictions = prediction_service.predict_multiple(items)

        result_df = df.copy()
        result_df["predicted_price"] = predictions

        result_csv_path = "predicted_prices.csv"
        result_df.to_csv(result_csv_path, index=False)

        return FileResponse(
            result_csv_path, media_type="text/csv", filename=result_csv_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке CSV файла: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
