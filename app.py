import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st

# Загружаем модель
with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)
class_names = config["class_names"]
IMG_SIZE = config["img_size"]
model = load_model("best_classification_model.h5")

st.set_page_config(page_title="Классификатор животных", layout="wide")
st.title("🐾 Классификатор животных")
st.write("Выбери изображение и нажми кнопку")

uploaded_file = st.file_uploader("Загрузи фото", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Твоё фото", width=250)

    if st.button("🔍 Классифицировать"):
        with st.spinner("Думаю..."):
            # Предобработка
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img_resized, dtype="float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            # Предсказание
            preds = model.predict(arr, verbose=0)[0]
            idx = np.argmax(preds)

            # Результат
            st.success(f"### 🎯 Это **{class_names[idx]}** ({(preds[idx]*100):.1f}%)")

            # График вероятностей
            fig, ax = plt.subplots(figsize=(6, 2))
            colors = ['green' if i == idx else 'gray' for i in range(len(class_names))]
            ax.barh(class_names, preds, color=colors)
            ax.set_xlim(0, 1)
            st.pyplot(fig)

            # Таблица
            for cls, prob in sorted(zip(class_names, preds), key=lambda x: x[1], reverse=True):
                st.write(f"- {cls}: {prob*100:.1f}%")