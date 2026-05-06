import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Адрес вашего API на Render
API_URL = "https://your-app.onrender.com/predict"

st.set_page_config(page_title="Animal Classifier", layout="wide")
st.title("🐾 Классификатор животных")
st.write("Загрузите изображение или нарисуйте его на холсте")

# Список классов
class_names = ["cat", "dog", "elephant", "horse", "lion"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Загрузка изображения")
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("🎨 Нарисуйте изображение")
    drawing_mode = "freedraw"
    stroke_width = st.slider("Толщина кисти", 1, 30, 10)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
    )

def predict_image(image):
    """Отправка изображения на API"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.png", img_byte_arr, "image/png")},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения: {e}")
        return None

def plot_probabilities(probabilities):
    """Визуализация распределения вероятностей"""
    fig, ax = plt.subplots(figsize=(8, 3))
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = plt.cm.viridis(np.array(probs) / max(probs))

    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel("Вероятность")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.2%}", va="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

# Обработка загруженного файла
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", width=300)

    if st.button("🔍 Классифицировать загруженное"):
        with st.spinner("Классификация..."):
            result = predict_image(image)
            if result:
                st.success(f"🎯 Предсказанный класс: **{result['predicted_class']}**")
                st.metric("Уверенность", f"{result['confidence']:.2%}")
                st.subheader("Распределение вероятностей")
                plot_probabilities(result['probabilities'])

# Обработка рисунка
if canvas_result.image_data is not None:
    drawn_image = Image.fromarray(
        canvas_result.image_data.astype("uint8")
    ).convert("RGB")

    if st.button("🔍 Классифицировать рисунок"):
        with st.spinner("Классификация..."):
            result = predict_image(drawn_image)
            if result:
                st.success(f"🎯 Предсказанный класс: **{result['predicted_class']}**")
                st.metric("Уверенность", f"{result['confidence']:.2%}")
                st.subheader("Распределение вероятностей")
                plot_probabilities(result['probabilities'])