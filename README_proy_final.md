## Proyecto: FútbolMarket

Valor de jugadores en el mercado del fútbol.

## Descripción del proyecto:

Este proyecto utiliza técnicas de machine learning para predecir el valor de mercado de jugadores de fútbol en función de variables como la edad y nacionalidad del jugador; la liga; equipo al cual pertenece; cantidad de goles, minutos jugados, tarjetas rojas y amarillas por partido; entre otros.

El proyecto se enmarca en el proyecto final del bootcamp de 4GeeksAcademy de Data Science and Machine Learning.

Autores: 
- Guazzelli, Victoria Laura.
- Mosquera, Facundo.

## Objetivo:

El objetivo es aplicar modelos de regresión y explorar la importancia de las variables para entender qué factores influyen en el precio de mercado de los jugadores de fútbol.

## Análisis.

La información se ha obtenido de Kaggle https://www.kaggle.com/datasets/davidcariboo/player-scores, donde se encuentran varios dataset con información de jugadores, de ligas, de partidos, que hubo que procesar para hacer un dataset final. 

### Preprocesamiento de los dataset:

- appearence.csv: Un dataset en el cual cada registro contiene información de cada jugador en cada partido de diferentes ligas desde el año 2012 al 2024. 

Dificultad: Gran cantidad de registros (más de 1.650.000) y su gran tamaño no permitía trabajarlo en codespace.

Solución: Hemos generado un nuevo dataset (appearence_agrupado.csv) que ha sido trabajado en el google colab en el siguiente link:
https://colab.research.google.com/drive/1kkoDlp4htYlhvd1pJ4Mg8GcqiJniWnwx#scrollTo=55sTvyXtUQs0

Cada registro de dicho dataset contiene la información de los partidos de cada jugador en cada liga, unificandolos en el mismo año (teniendo en cuenta la edad del jugador). 
A continuación describimos el procedimiento:

1. Selección de columnas relevantes en el dataset appearence:
'game_id': identificación de cada partido, 
'player_id': identificación de jugador, 
'player_club_id': identificación del club del jugador, 
'date': fecha del partido, 
'player_name': nombre del jugador, 
'competition_id': identificación de la liga, 
'yellow_cards': tarjetas amarillas del jugador en el partido, 
'red_cards': tarjetas rojas del jugador en el partido, 
'goals': cantidad de goles del jugador en el partido, 
'assists': cantidad de asistencias del jugador en el partido, 
'minutes_played': cantidad de minutos jugador por jugador en el partido.

2. Adición de columnas de otros dataset (Herramienta de pandas: merge):
'club_name': nombre del club del jugador, 
'date_of_birth': fecha de nacimiento del jugador (para poder calcular la edad en cada partido),
'age': edad del jugador en el partido.

3. Agrupación de filas por jugador, año y liga (Herramienta: groupby). Generando:
'matches_played': cantidad de partidos jugados (de cada jugador en cada liga y en cada año),
'yellow_cards': promedio de tarjetas amarillas,
'red_cards': promedio de tarjetas rojas,
'goals': promedio de goles,
'assists': promedio de asistencias,
'minutes_played': promedio de minutos jugados,
"age": edad del jugador. 

