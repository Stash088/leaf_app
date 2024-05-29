import streamlit as st
from PIL import Image
import numpy as np
import cv2
import utils
import imutils
import base64
import pandas as pd
from matplotlib import pyplot as plt
st.set_page_config(page_title="Plant Leaf Analysis", page_icon=":seedling:", layout="wide")
st.title("Приложение для фенотипирования ")
uploaded_file = st.file_uploader("Загрузка изображения", type=["jpg", "jpeg", "png"])
st.sidebar.title('Навигация')
section = st.sidebar.radio('Перейти к разделу:', ('Основные показатели', 'Спектральный анализ'))


if section == 'Спектральный анализ':
    st.header('Спектральный анализ')
    st.write('Это содержимое раздела "Спектральный анализ".')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        fractal_sphericity = utils.fractal_sphericity(image_rgb)
        texture_index = utils.fractal_texture_index(image_rgb)
        chla, chlb, chla_b = utils.calculate_chlorophyll_index(image_rgb)
        ndvi = utils.ndvi_image(image_rgb)
        st.image(image, caption="Оригинальное изображение ")
        st.image(chla, caption='Индекс хлорофилла a (Chla)', use_column_width=True , clamp=True)
        st.image(chlb, caption='Индекс хлорофилла b (Chlb)', use_column_width=True , clamp=True)
        fig, ax = plt.subplots()
        ax.imshow(ndvi, cmap='RdYlGn')
        ax.axis('off')
        st.pyplot(fig)
        st.write("Фрактальный индекс текстуры : ", round(texture_index, 3))
        st.write("фрактальная сферичности : ", round(fractal_sphericity, 3))


elif section == 'Основные показатели':
    st.header('Основные характеристики')
    st.write('Это содержимое раздела "Другие характеристики".')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        _, _, assymetry = utils.measure_extract(image_rgb)
        edge_equalized = utils.canny_edges(image_rgb)
        class_leaf = utils.classfier_leaf(image_rgb)
        venation_leaf = utils.venation_leaf(image_rgb)
        length,width = utils.calculate_shapes(image_rgb)
        area = utils.calculate_area_leafs(image_rgb)
        data = {
            'Длина см': [round(length, 3)*100],
            'Ширина см': [round(width, 3)*100],
            'Площадь см^2': [round(area, 3)],
            'Флуктуирующая асимметрия листа': [round(assymetry, 3)],
            'Класс листа': [class_leaf],
            'Венозность листа': [venation_leaf]
        }
        df = pd.DataFrame(data)
        st.image(image, caption="Оригинальное изображение ")
        st.image(edge_equalized, caption="Изображение после детектора Canny")
        st.table(df)