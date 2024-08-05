import streamlit as st
from transformers import AutoImageProcessor, ResNetForImageClassification
from procesos.texto_a_imagen import Texto_a_Imagen
from procesos.imagen_a_texto import Imagen_a_texto
import torch
from PIL import Image
import numpy as np

# Cargar modelos
resnet50_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Función para clasificar imagen con ResNet50
def clasifica_imagen(imagen):
    imagen = imagen.convert("RGB")
    imagen = np.array(imagen) / 255.0
    imagen = torch.tensor(imagen).unsqueeze(0)
    outputs = resnet50_model(imagen)
    return outputs

# configurar ancho
st.set_page_config(layout="wide")
# Crear interfaz de usuario con Streamlit
st.title("Aplicación de Generación y Clasificación de Imágenes")
# configurar columnas
col1, col2, col3 = st.columns(3,gap="large")

with col1:
    # Sección de Generación de Imágenes
    st.header("Generación de Imágenes")
    texto_ingresado = st.text_input("Ingrese un texto para generar una imagen")
    obj_txt_img = Texto_a_Imagen()
    obj_txt_img.carga_modelo()
    if st.button("Generar"):
        imagen = obj_txt_img.generar(texto_ingresado,negative_prompt="low quality, ugly",)
        st.image(imagen)

with col2:
    # Sección de Clasificación de Imágenes
    st.header("Clasificación de Imágenes")
    archivo_subido = st.file_uploader("Subir una imagen")
    obj_img_txt = Imagen_a_texto()
    obj_img_txt.carga_modelo()
    if archivo_subido is not None:
        imagen = Image.open(archivo_subido)
        respuesta = obj_img_txt.clasificar(imagen)
        st.write(respuesta)

with col3:
    st.header("Generar y Clasificar")
    # Conectar la imagen generada con la clasificación
    texto_ingresado = st.text_input("Ingrese un texto para generar una imagen y clasificarlo")
    obj_txt_img = Texto_a_Imagen()
    obj_img_txt = Imagen_a_texto()
    obj_txt_img.carga_modelo()
    obj_img_txt.carga_modelo()
    if st.button("Generar y Clasificar"):
        imagen = obj_txt_img.generar(texto_ingresado,negative_prompt="low quality, ugly",)
        respuesta = obj_img_txt.clasificar(st.image(imagen))
        st.write(respuesta)