import pandas as pd
import streamlit as st
import numpy as np
#import plotly.express as px
import gzip
import joblib
import os
from pickle import load

#print(os.getcwd())

label_encoder_city_of_birth=load(open("./src/le_city_of_birth.sav", "rb"))
label_encoder_country_of_birth=load(open("./src/le_country_of_birth.sav", "rb"))
label_encoder_competition_id=load(open("./src/le_competition_id.sav", "rb"))
label_encoder_club_name=load(open("./src/le_club_name.sav", "rb"))
label_encoder_foot=load(open("./src/le_foot.sav", "rb"))
label_encoder_sub_position=load(open("./src/le_sub_position.sav", "rb"))
label_encoder_position=load(open("./src/le_position.sav", "rb"))

def cargar_modelo_comprimido(ruta):
    """Carga el modelo comprimido con gzip."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo {ruta} no existe. Verifica que está en la carpeta correcta.")
    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    return modelo

# Define la ruta del modelo comprimido
RUTA_MODELO = "./src/modelo.pkl.gz"

# Intenta cargar el modelo
try:
    st.write("Cargando el modelo...")
    model = cargar_modelo_comprimido(RUTA_MODELO)
    st.success("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()  # Detiene la app si no se puede cargar el modelo


@st.cache(persist=True)
def load_data():
    df = pd.read_csv("./src/df_new.csv")
    return df

# Solo se ejecutará una vez si ya está en caché
data = load_data()
st.markdown('<style>description{color:blue;}</style>', unsafe_allow_html=True)
st.title('Prediccin de el valor de mercado de un jugador de fútbol')
st.markdown("<description> Descripcion a gusto </description>", unsafe_allow_html=True)
# Agregar un título a la barra lateral
st.sidebar.title('Selecciona los parámetros para analizar la predicción de el valor de mercado de un futbolista')

# Agregar otros elementos a la barra lateral según sea necesario
# Por ejemplo, un control deslizante:
matches_played = st.sidebar.slider('matches_played', 0, 100, 25)
yellow_cards = st.sidebar.slider('yellow_cards', 0, 100, 25)
red_cards = st.sidebar.slider('red_cards', 0, 100, 25)
goals = st.sidebar.slider('goals', 0, 100, 25)
assists = st.sidebar.slider('assists', 0, 100, 25)
minutes_played = st.sidebar.slider('minutes_played', 0, 100, 25)
age = st.sidebar.slider('age', 0, 100, 25)
height_in_cm = st.sidebar.slider('height_in_cm', 0, 100, 25)
highest_market_value_in_eur = st.sidebar.slider('highest_market_value_in_eur', 0, 100, 25)
club_name = st.selectbox('Select a club name', data['club_name'])
foot = st.selectbox('Select a foot', data['foot'])
position = st.selectbox('Select a position', data['position'])
sub_position = st.selectbox('Select a sub position', data['sub_position'])
country_of_birth = st.selectbox('Select a country of birth', data['country_of_birth'])
city_of_birth= st.selectbox('Select a city of birth', data['city_of_birth'])
competition_id = st.selectbox('Select a competition', data['competition_id'])


competition_id_le = label_encoder_competition_id.transform([competition_id])[0]
club_name_le = label_encoder_club_name.transform([club_name])[0]
foot_le = label_encoder_foot.transform([foot])[0]
position_le = label_encoder_position.transform([position])[0]
sub_position_le = label_encoder_sub_position.transform([sub_position])[0]
country_of_birth_le = label_encoder_country_of_birth.transform([country_of_birth])[0]
city_of_birth_le = label_encoder_city_of_birth.transform([city_of_birth])[0]


info=[matches_played, yellow_cards, red_cards, goals, assists, minutes_played, age, height_in_cm, highest_market_value_in_eur, competition_id_le, club_name_le, foot_le, position_le, sub_position_le, country_of_birth_le,  city_of_birth_le]


if st.button("Realizar predicción"):
    try:
        # Supongamos que el modelo tiene un método predict
        prediccion = model.predict(info)[0]
        st.write(f"Predicción del modelo: {prediccion}")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
