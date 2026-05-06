import io
import pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Загрузка конфигурации и модели при старте
with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

class_names = config["class_names"]
IMG_SIZE = config["img_size"]

model = load_model("best_classification_model.h5")
print("Модель загружена")

app = FastAPI(title="Animal Classifier API", version="1.0")

# Разрешаем CORS для Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    """Предобработка изображения"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def root():
    return {"message": "Animal Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Чтение и предобработка
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Предсказание
    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])

    # Формирование ответа
    probabilities = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities
    }