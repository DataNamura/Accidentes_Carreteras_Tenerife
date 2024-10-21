# --------------------------- #
#      1. Importaciones        #
# --------------------------- #

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import requests
import osmnx as ox
import streamlit.components.v1 as components
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
from datetime import datetime
import locale
# --------------------------- #
# 2. Cargar Variables de Entorno #
# --------------------------- #

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener la clave de API de Google desde las variables de entorno
API_KEY_GOOGLE = st.secrets["GOOGLE_API_KEY"]


if not API_KEY_GOOGLE:
    st.error("⚠️ La clave de API de Google no está configurada. Por favor, establece la variable de entorno 'GOOGLE_API_KEY'.")
    st.stop()

# --------------------------- #
#     3. Configuración Página    #
# --------------------------- #


# Configuración de la página
st.set_page_config(page_title="🚗 Análisis de Accidentes en Carreteras", layout="wide")

# --------------------------- #
#      INTRODUCCIÓN E INSTRUCCIONES #
# --------------------------- #

# 🚗 Análisis y Predicción de Accidentes en Carreteras de Tenerife

st.markdown("""
# 🚗 Análisis y Predicción de Accidentes en Carreteras de Tenerife

Esta aplicación interactiva te permite analizar accidentes de tráfico en las carreteras de Tenerife entre **2010 y 2024**. Utiliza datos históricos y tecnología avanzada para visualizar mapas, obtener información en tiempo real sobre el tráfico, y predecir la probabilidad de accidentes en diferentes tramos de carretera.
""")

# Desplegable para las características principales
with st.expander("🔍 Características principales"):
    st.markdown("""
    - **Análisis de accidentes**: Explora los accidentes de tráfico por carretera, hora del día, día de la semana, y más.
    - **Predicción en tiempo real**: Obtén una predicción sobre la probabilidad de que ocurra un accidente en las condiciones actuales.
    - **Visualización de tráfico en tiempo real**: Muestra el tráfico actual en las carreteras seleccionadas usando Google Maps.
    """)

# Desplegable para la barra lateral de filtros
with st.expander("📊 Cómo usar la aplicación"):
    st.markdown("""
    ### Filtros en la barra lateral:
    1. **📅 Selección de años**: 
       - Usa el control deslizante para seleccionar el rango de años que te interesa analizar. 
       - El valor predeterminado incluye todos los años disponibles, pero puedes ajustar el rango a tu gusto.

    2. **🛣️ Selección de carreteras**: 
       - Puedes seleccionar una o varias carreteras para analizar accidentes en esas zonas. Si seleccionas "Seleccionar Todas", verás los datos de todas las carreteras disponibles.

    3. **⏰ Filtrar por hora del día**: 
       - Puedes ajustar el análisis para ver accidentes que ocurrieron en un rango de horas específico del día. Esto es útil si quieres analizar solo las horas pico o un periodo en particular.
       
    Los filtros que seleccionas aquí determinarán los resultados que verás en las siguientes secciones de gráficos y mapas.
    """)

# Desplegable para la sección de mapas
with st.expander("🗺️ Sección de Mapas"):
    st.markdown("""
    ### 1. Mapa de tráfico en tiempo real:
    - En esta sección, puedes visualizar el tráfico actual en las carreteras de Tenerife gracias a la integración con Google Maps. 
    - El mapa te muestra en tiempo real dónde hay más tráfico, con una leyenda que indica el nivel de tráfico.

    ### 2. Mapa interactivo de carreteras seleccionadas:
    - Este mapa muestra las carreteras que seleccionaste en la barra lateral de forma gráfica.

# Desplegable para la sección de gráficos
with st.expander("📊 Sección de Gráficos"):
    st.markdown("""
    ### Cómo funciona la sección de gráficos:
    - Después de aplicar los filtros en la barra lateral (años, carreteras y horas), los gráficos te mostrarán un análisis detallado sobre los accidentes filtrados.
    - Los gráficos incluirán información como:
      - **Accidentes por tramo**: Muestra los tramos con más accidentes en las carreteras seleccionadas.
      - **Accidentes por hora del día**: Te permite ver en qué horas del día ocurren más accidentes.
      - **Accidentes por día de la semana**: Muestra los días de la semana con más accidentes.
      - **Accidentes por mes**: Analiza cómo varía la siniestralidad a lo largo del año.

    ### Qué hacer en esta sección:
    - Una vez aplicados los filtros, los gráficos se actualizarán automáticamente.
    - Cada gráfico proporciona insights clave, por ejemplo:
      - **¿A qué horas ocurren más accidentes?**
      - **¿Cuáles son los días más peligrosos en las carreteras seleccionadas?**
    - Puedes utilizar esta información para identificar patrones y tendencias.

    ### Descarga de informes:
    - Todos los gráficos que se generan se pueden descargar en formato PNG, lo cual es útil para crear informes o presentaciones.
    """)

# Desplegable para la sección de predicción
with st.expander("🔮 Sección de Predicción"):
    st.markdown("""
    ### Predicción de accidentes:
    - La aplicación te ofrece una estimación de la probabilidad de que ocurra un accidente en tiempo real. Esta predicción se basa en datos históricos y factores como el día de la semana, la hora, y el tramo de carretera que seleccionaste.

    ### Cómo funciona:
    - Selecciona una carretera y un tramo en la barra lateral, y la aplicación calculará la probabilidad de que ocurra un accidente bajo las condiciones actuales (hora y día).
    
    ### Barómetro de riesgo:
    - La predicción se muestra visualmente como un barómetro, indicando si el riesgo de accidente es bajo (verde), moderado (naranja), o alto (rojo).
    """)


# --------------------------- #
#     4. Cargar Datos           #
# --------------------------- #

@st.cache_data
def load_data():
    file_id = '1R4dx-DCwZ29fNvdsT-JHorZPc3LnfT0p'  # ID del archivo en Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'incidencias_full_limpio.csv'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    df['incidencia_fecha_inicio'] = pd.to_datetime(df['incidencia_fecha_inicio'])
    df['hora'] = df['incidencia_fecha_inicio'].dt.hour
    df['dia_semana'] = df['incidencia_fecha_inicio'].dt.dayofweek
    df['mes'] = df['incidencia_fecha_inicio'].dt.month
    return df

df = load_data()

# --------------------------- #
#     5. Definición Funciones    #
# --------------------------- #

# Función para obtener coordenadas a partir de un nombre o dirección usando la API de Geocoding
def obtener_coordenadas_lugar(lugar, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': lugar, 'key': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if len(data['results']) > 0:
            lat = data['results'][0]['geometry']['location']['lat']
            lon = data['results'][0]['geometry']['location']['lng']
            return f"{lat},{lon}"
        else:
            st.error(f"No se encontró el lugar: {lugar}")
            return None
    else:
        st.error(f"Error al obtener coordenadas: {response.status_code}")
        return None

# Función para obtener la ruta desde Google Maps Directions API
def obtener_ruta_google_maps(origen, destino, api_key):
    url = f"https://maps.googleapis.com/maps/api/directions/json"
    params = {
        'origin': origen,
        'destination': destino,
        'key': api_key,
        'departure_time': 'now',
        'traffic_model': 'best_guess',
        'mode': 'driving'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error al obtener la ruta: {response.status_code}")
        return None

# Función para mostrar la ruta en el mapa y datos de tráfico
def mostrar_ruta_en_mapa_con_trafico(ruta):
    if ruta and 'routes' in ruta and len(ruta['routes']) > 0:
        tiempo_sin_trafico = ruta['routes'][0]['legs'][0]['duration']['value'] / 60
        tiempo_con_trafico = ruta['routes'][0]['legs'][0]['duration_in_traffic']['value'] / 60
        retraso = tiempo_con_trafico - tiempo_sin_trafico

        st.subheader("📊 Datos de Tráfico")
        st.write(f"Tiempo estimado sin tráfico: **{tiempo_sin_trafico:.2f} minutos**")
        st.write(f"Tiempo estimado con tráfico: **{tiempo_con_trafico:.2f} minutos**")
        st.write(f"Retraso estimado debido al tráfico: **{retraso:.2f} minutos**")

        start_location = ruta['routes'][0]['legs'][0]['start_location']
        mapa = folium.Map(location=[start_location['lat'], start_location['lng']], zoom_start=13)

        points = []
        for step in ruta['routes'][0]['legs'][0]['steps']:
            lat = step['start_location']['lat']
            lng = step['start_location']['lng']
            points.append([lat, lng])

        folium.PolyLine(points, color="blue", weight=5, opacity=0.8).add_to(mapa)
        return mapa
    else:
        st.warning("No se encontró una ruta válida.")
        return None



# --------------------------- #
#        6. Mapa de Tráfico     #
# --------------------------- #


# Título de la sección del mapa
st.title("📍 Google Maps con Tráfico en Tiempo Real")

# Obtener la API Key desde los secretos
api_key = st.secrets["GOOGLE_API_KEY"]

# Verificar que la API Key esté configurada
if not api_key:
    st.error("⚠️ La clave de API de Google no está configurada. Por favor, establece la variable de entorno 'GOOGLE_API_KEY'.")
    st.stop()

# HTML dinámico con la API Key inyectada
html_content = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>Google Maps con Tráfico en Tiempo Real</title>
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
    <style>
      /* Estilo para la leyenda */
      .legend {{
        background: white;
        padding: 10px;
        margin: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
        font-size: 12px;
      }}
      .legend div {{
        display: flex;
        align-items: center;
        margin-bottom: 5px;
      }}
      .color-box {{
        width: 20px;
        height: 20px;
        margin-right: 5px;
      }}
      .green {{
        background-color: green;
      }}
      .yellow {{
        background-color: yellow;
      }}
      .red {{
        background-color: red;
      }}
    </style>
    <script>
      function initMap() {{
        var mapOptions = {{
          zoom: 8,
          center: {{ lat: 28.3000, lng: -16.5000 }}
        }};
        var map = new google.maps.Map(document.getElementById('map'), mapOptions);

        // Añadir la capa de tráfico
        var trafficLayer = new google.maps.TrafficLayer();
        trafficLayer.setMap(map);

        // Crear la leyenda manualmente
        var legend = document.createElement('div');
        legend.innerHTML = `
          <div class="legend">
            <div><div class="color-box green"></div>Tráfico Fluido</div>
            <div><div class="color-box yellow"></div>Tráfico Moderado</div>
            <div><div class="color-box red"></div>Tráfico Denso</div>
          </div>
        `;

        // Colocar la leyenda en la parte inferior derecha
        map.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
      }}
    </script>
  </head>
  <body>
    <div id="map" style="height: 600px; width: 100%;"></div>
  </body>
</html>
"""

# Renderizar el contenido HTML en Streamlit
components.html(html_content, height=600, scrolling=True)


# --------------------------- #
#      7. Procesamiento Datos  #
# --------------------------- #



# --------------------------- #
# 8. Análisis y Predicción      #
# --------------------------- #

# Filtros en la barra lateral
st.sidebar.header("🔎 Filtros")

# Subtítulo y descripción para el filtro de años
st.sidebar.subheader("📅 Rango de Años")
st.sidebar.markdown("Selecciona el rango de años que deseas analizar.")
min_year = int(df['annio'].min())
max_year = int(df['annio'].max())
años = st.sidebar.slider("Selecciona el Rango de Años", min_year, max_year, (min_year, max_year))

# Separador para organizar los filtros
st.sidebar.markdown("---")

# Subtítulo y filtro para seleccionar carreteras
st.sidebar.subheader("🛣️ Selección de Carreteras")
st.sidebar.markdown("Elige una o varias carreteras para filtrar los datos.")
carreteras_disponibles = df['carretera_nombre'].unique().tolist()
carreteras_disponibles.insert(0, "Seleccionar Todas")
carreteras_seleccionadas = st.sidebar.multiselect(
    "Carreteras",
    carreteras_disponibles,
    default=["Seleccionar Todas"]
)

# Separador
st.sidebar.markdown("---")

# Subtítulo y descripción para el filtro de horas
st.sidebar.subheader("⏰ Hora del Día")
st.sidebar.markdown("Filtra los datos según la hora del día en la que ocurrieron los incidentes.")
hora = st.sidebar.slider("Selecciona la Hora del Día", 0, 23, (0, 23))

# Separador
st.sidebar.markdown("---")

# Filtrar el dataframe por años, carreteras y hora seleccionados
if "Seleccionar Todas" in carreteras_seleccionadas:
    carreteras_seleccionadas = df['carretera_nombre'].unique().tolist()

df_filtrado = df[
    (df['annio'] >= años[0]) & (df['annio'] <= años[1]) &
    (df['carretera_nombre'].isin(carreteras_seleccionadas)) &
    (df['hora'] >= hora[0]) & (df['hora'] <= hora[1])
]

# Filtrar solo los accidentes
df_filtrado_accidentes = df_filtrado[df_filtrado['es_accidente'] == 'Accidente']

# Lista para almacenar los insights
insights = []

# Configuración global del estilo del gráfico
sns.set_style("darkgrid")
plt.style.use("dark_background")

# --------------------------- #
#          9. Gráficos        #
# --------------------------- #

def crear_grafico_1(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 1: Número de Accidentes por Tramo"""
    st.header("🔝 1. Número de Accidentes en las Carreteras Seleccionadas")
    
    accidentes_por_tramo = df_filtrado_accidentes['tramo_nombre'].value_counts().head(10)
    
    if accidentes_por_tramo.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No hay suficientes datos', horizontalalignment='center', verticalalignment='center', fontsize=12, color='white')
        plt.xticks([])
        plt.yticks([])
        plt.title("🚧 Top 10 Tramos de Carretera con Más Accidentes")
        plt.tight_layout()
        st.pyplot(plt)
    else:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=accidentes_por_tramo.values, y=accidentes_por_tramo.index, palette="Blues_r")
        plt.title("🚧 Top 10 Tramos de Carretera con Más Accidentes")
        plt.xlabel("Número de Accidentes")
        plt.ylabel("Tramo")
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_accidentes_por_tramo.png",
            mime="image/png"
        )
        
        # Tabla con detalles de los Top 10 Tramos (ahora desplegable)
        with st.expander("📋 Ver detalles de Top 10 Tramos de Carretera con Más Accidentes"):
            top_tramos_df = accidentes_por_tramo.reset_index()
            top_tramos_df.columns = ['Tramo', 'Número de Accidentes']
            top_tramos_df['Porcentaje (%)'] = (top_tramos_df['Número de Accidentes'] / total_accidentes_filtrado) * 100
            st.table(top_tramos_df.style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        top_tramo = accidentes_por_tramo.index[0]
        top_accidentes = accidentes_por_tramo.iloc[0]
        porcentaje_top_tramo = (top_accidentes / total_accidentes_filtrado) * 100
        insight_1 = f"""
        ### 💡 Insight 1: Tramo con Mayor Número de Accidentes
        El tramo con más accidentes es **{top_tramo}**, con **{top_accidentes} accidentes**, lo que representa un **{porcentaje_top_tramo:.2f}%** del total de accidentes en el rango seleccionado.
        """
        st.markdown(insight_1)
        insights.append(insight_1)


def crear_grafico_2(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 2: Distribución de Accidentes por Hora del Día"""
    st.header("⏱️ 2. Distribución de Accidentes por Hora del Día")
    
    accidentes_por_hora = df_filtrado_accidentes['hora'].value_counts().sort_index()
    todas_las_horas = pd.DataFrame({'hora': range(24)})
    accidentes_por_hora_df = accidentes_por_hora.reset_index()
    accidentes_por_hora_df.columns = ['hora', 'numero_accidentes']
    accidentes_por_hora_df = todas_las_horas.merge(accidentes_por_hora_df, how='left', on='hora')
    accidentes_por_hora_df['numero_accidentes'] = accidentes_por_hora_df['numero_accidentes'].fillna(0).astype(int)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hora', y='numero_accidentes', data=accidentes_por_hora_df, marker='o', color="skyblue")
    plt.title("⏰ Accidentes por Hora del Día")
    plt.xlabel("Hora del Día")
    plt.ylabel("Número de Accidentes")
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    
    # Botón de descarga del gráfico
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label="💾 Descargar gráfico",
        data=buf,
        file_name="grafico_accidentes_por_hora.png",
        mime="image/png"
    )
    
    # Tabla con detalles de accidentes por hora (ahora desplegable)
    with st.expander("📋 Ver detalles de Accidentes por Hora del Día"):
        st.table(accidentes_por_hora_df.rename(columns={'numero_accidentes': 'Número de Accidentes'}).style.format({'Número de Accidentes': '{:d}'}))
    
    # Insight
    hora_pico = accidentes_por_hora_df.loc[accidentes_por_hora_df['numero_accidentes'].idxmax(), 'hora']
    num_accidentes_hora = accidentes_por_hora_df['numero_accidentes'].max()
    porcentaje_hora_pico = (num_accidentes_hora / total_accidentes_filtrado) * 100
    insight_2 = f"""
    ### 💡 Insight 2: Hora Pico de Accidentes
    La hora con más accidentes es a las **{hora_pico}:00 horas**, con **{num_accidentes_hora} accidentes**, representando un **{porcentaje_hora_pico:.2f}%** del total de accidentes. Este pico horario puede indicar mayor tráfico.
    """
    st.markdown(insight_2)
    insights.append(insight_2)


def crear_grafico_3(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 3: Correlación entre Día de la Semana y Número de Accidentes"""
    st.header("📅 3. Correlación entre Día de la Semana y Número de Accidentes")
    correlacion_df = df_filtrado_accidentes.groupby('dia_semana').size().reset_index(name='accidentes')
    
    dias_semana_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    if correlacion_df.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='dia_semana', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
        plt.title('📊 Correlación entre Día de la Semana y Número de Accidentes')
        plt.xlabel('Día de la Semana')
        plt.ylabel('Número de Accidentes')
        plt.xticks(ticks=correlacion_df['dia_semana'], labels=dias_semana_nombres, rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_correlacion_dia_semana_accidentes.png",
            mime="image/png"
        )
        
        # Tabla con detalles de accidentes por día de la semana (ahora desplegable)
        with st.expander("📋 Ver detalles de Accidentes por Día de la Semana"):
            correlacion_df_display = correlacion_df.copy()
            correlacion_df_display['Día de la Semana'] = correlacion_df_display['dia_semana'].apply(lambda x: dias_semana_nombres[x])
            correlacion_df_display = correlacion_df_display.rename(columns={'accidentes': 'Número de Accidentes'})
            correlacion_df_display['Porcentaje (%)'] = (correlacion_df_display['Número de Accidentes'] / total_accidentes_filtrado) * 100
            st.table(correlacion_df_display[['Día de la Semana', 'Número de Accidentes', 'Porcentaje (%)']].style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        dia_semana_pico = correlacion_df_display.loc[correlacion_df_display['Número de Accidentes'].idxmax(), 'dia_semana']
        dia_semana_nombre = dias_semana_nombres[dia_semana_pico]
        num_accidentes_dia = correlacion_df_display['Número de Accidentes'].max()
        porcentaje_dia_pico = (num_accidentes_dia / total_accidentes_filtrado) * 100
        
        insight_3 = f"""
        ### 💡 Insight 3: Día con Más Accidentes
        El día con más accidentes es el **{dia_semana_nombre}**, con **{num_accidentes_dia} accidentes**, representando un **{porcentaje_dia_pico:.2f}%** del total de accidentes. Esto sugiere que los **{dia_semana_nombre}** tienen una mayor siniestralidad.
        """
        st.markdown(insight_3)
        insights.append(insight_3)


def crear_grafico_4(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 4: Correlación entre Mes y Número de Accidentes"""
    st.header("📆 4. Correlación entre Mes y Número de Accidentes")
    correlacion_df_mes = df_filtrado_accidentes.groupby('mes').size().reset_index(name='accidentes')
    
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    if correlacion_df_mes.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='mes', y='accidentes', data=correlacion_df_mes, scatter_kws={'s':50}, line_kws={'color':'blue'})
        plt.title('📈 Correlación entre Mes y Número de Accidentes')
        plt.xlabel('Mes')
        plt.ylabel('Número de Accidentes')
        plt.xticks(ticks=correlacion_df_mes['mes'], labels=meses_nombres, rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_correlacion_mes_accidentes.png",
            mime="image/png"
        )
        
        # Tabla con detalles de accidentes por mes (ahora desplegable)
        with st.expander("📋 Ver detalles de Accidentes por Mes"):
            correlacion_df_mes_display = correlacion_df_mes.copy()
            correlacion_df_mes_display['Mes'] = correlacion_df_mes_display['mes'].apply(lambda x: meses_nombres[x-1])
            correlacion_df_mes_display = correlacion_df_mes_display.rename(columns={'accidentes': 'Número de Accidentes'})
            correlacion_df_mes_display['Porcentaje (%)'] = (correlacion_df_mes_display['Número de Accidentes'] / total_accidentes_filtrado) * 100
            st.table(correlacion_df_mes_display[['Mes', 'Número de Accidentes', 'Porcentaje (%)']].style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        mes_pico = correlacion_df_mes_display.loc[correlacion_df_mes_display['Número de Accidentes'].idxmax(), 'mes']
        mes_nombre = meses_nombres[int(mes_pico) - 1]
        num_accidentes_mes = correlacion_df_mes_display['Número de Accidentes'].max()
        porcentaje_mes_pico = (num_accidentes_mes / total_accidentes_filtrado) * 100
        
        insight_4 = f"""
        ### 💡 Insight 4: Mes con Más Accidentes
        El mes con más accidentes es **{mes_nombre}**, con **{num_accidentes_mes} accidentes**, representando un **{porcentaje_mes_pico:.2f}%** del total de accidentes. Esto podría estar relacionado con condiciones climáticas o aumento de tráfico en ese mes.
        """
        st.markdown(insight_4)
        insights.append(insight_4)


def crear_grafico_5(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 5: Mapa de Calor de Accidentes por Día de la Semana y Hora"""
    st.header("🔥 5. Mapa de Calor: Accidentes por Día de la Semana y Hora del Día")
    heatmap_data = df_filtrado_accidentes.groupby(['dia_semana', 'hora']).size().unstack()
    
    dias_semana_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    if heatmap_data.empty or heatmap_data.shape[1] == 0:
        st.write("⚠️ No hay suficientes datos para mostrar el mapa de calor.")
    else:
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="Blues_r", annot=True, fmt=".0f")
        plt.title("🌡️ Accidentes por Día de la Semana y Hora del Día")
        plt.xlabel("Hora del Día")
        plt.ylabel("Día de la Semana")
        plt.yticks(ticks=[0,1,2,3,4,5,6], labels=dias_semana_nombres, rotation=0)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_mapa_calor_accidentes.png",
            mime="image/png"
        )
        
        # Tabla con detalles de accidentes por día y hora (ahora desplegable)
        with st.expander("📋 Ver detalles de Accidentes por Día de la Semana y Hora del Día"):
            heatmap_display_df = heatmap_data.copy()
            heatmap_display_df.index = heatmap_display_df.index.map(lambda x: dias_semana_nombres[x])
            heatmap_display_df.columns = [f"{hora}:00" for hora in heatmap_display_df.columns]
            st.table(heatmap_display_df.style.format("{:.0f}"))
        
        # Insight
        if heatmap_data.values.max() > 0:
            max_value = heatmap_data.values.max()
            max_indices = np.where(heatmap_data.values == max_value)
            dia_semana_pico = max_indices[0][0]
            hora_pico = heatmap_data.columns[max_indices[1][0]]
            dia_semana_nombre = dias_semana_nombres[dia_semana_pico]
            porcentaje_momento_critico = (max_value / total_accidentes_filtrado) * 100
            
            insight_5 = f"""
            ### 💡 Insight 5: Momento Crítico de Accidentes
            El mayor número de accidentes ocurre el **{dia_semana_nombre}** a las **{hora_pico}:00 horas**, con **{int(max_value)} accidentes**, representando un **{porcentaje_momento_critico:.2f}%** del total. Este es el momento más crítico en términos de siniestralidad.
            """
            st.markdown(insight_5)
            insights.append(insight_5)
        else:
            st.markdown("### 💡 Insight 5: Momento Crítico de Accidentes\nNo hay suficientes datos para determinar el momento crítico.")


def crear_grafico_6(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 6: Comparación de Accidentes por Carretera a lo Largo del Tiempo"""
    st.header("📉 6. Comparación de Accidentes por Carretera a lo Largo del Tiempo")
    
    # Filtro para seleccionar múltiples carreteras para el gráfico comparativo
    carreteras_disponibles_grafico = df['carretera_nombre'].unique().tolist()
    carreteras_seleccionadas_default = ['TF-1']  # Carretera por defecto
    
    carreteras_seleccionadas_grafico = st.multiselect(
        "🚧 Selecciona una o más Carreteras para Comparar",
        carreteras_disponibles_grafico,
        default=carreteras_seleccionadas_default
    )
    
    # Filtrar el dataframe para las carreteras seleccionadas
    df_filtrado_comparativo = df_filtrado_accidentes[df_filtrado_accidentes['carretera_nombre'].isin(carreteras_seleccionadas_grafico)]
    
    # Agrupación y visualización
    accidentes_por_carretera = df_filtrado_comparativo.groupby(['annio', 'carretera_nombre']).size().unstack()
    
    if accidentes_por_carretera.empty or accidentes_por_carretera.shape[1] == 0:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        plt.figure(figsize=(14, 8))  # Aumenta el tamaño de la figura
        accidentes_por_carretera.plot(kind='line', colormap="Blues_r", linewidth=2, marker='o', ax=plt.gca())
    
        # Ajustar la leyenda a la derecha del gráfico
        plt.legend(title='Carreteras', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
        plt.title("📈 Comparación de Accidentes a lo Largo del Tiempo")
        plt.xlabel("Año")
        plt.ylabel("Número de Accidentes")
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustar el espacio para la leyenda
    
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_comparativo_accidentes_carretera.png",
            mime="image/png"
        )
        
        # Tabla con detalles de accidentes por carretera (ahora desplegable)
        with st.expander("📋 Ver detalles de Accidentes por Carretera"):
            accidentes_por_carretera_df = accidentes_por_carretera.reset_index()
            accidentes_por_carretera_df = accidentes_por_carretera_df.melt(id_vars='annio', var_name='Carretera', value_name='Número de Accidentes')
            accidentes_por_carretera_df['Porcentaje (%)'] = (accidentes_por_carretera_df['Número de Accidentes'] / total_accidentes_filtrado) * 100
            st.table(accidentes_por_carretera_df.style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        total_accidentes_carreteras = df_filtrado_comparativo['carretera_nombre'].value_counts()
        if not total_accidentes_carreteras.empty:
            carretera_pico = total_accidentes_carreteras.idxmax()
            num_accidentes_carretera = total_accidentes_carreteras.max()
            insight_6 = f"""
            ### 💡 Insight 6: Carretera con Más Accidentes
            De las carreteras seleccionadas, la **{carretera_pico}** tiene el mayor número de accidentes con **{num_accidentes_carretera} accidentes** en el rango seleccionado.
            """
            st.markdown(insight_6)
            insights.append(insight_6)
        else:
            st.markdown("### 💡 Insight 6: Carretera con Más Accidentes\nNo hay suficientes datos para determinar la carretera con más accidentes.")


def crear_grafico_7(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 7: Distribución de Tipos de Incidencias (%)"""
    st.header("📊 7. Distribución de Tipos de Incidencias (%)")
    
    # Calcular el porcentaje de cada tipo de incidencia en el DataFrame filtrado
    tipo_incidencias_porcentaje = df_filtrado['incidencia_tipo'].value_counts(normalize=True) * 100
    
    if tipo_incidencias_porcentaje.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=tipo_incidencias_porcentaje.index, y=tipo_incidencias_porcentaje.values, palette="Blues_r")
        plt.title('📊 Distribución de Tipos de Incidencias (%)', fontsize=16)
        plt.xlabel('Tipo de Incidencia', fontsize=14)
        plt.ylabel('Porcentaje (%)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_distribucion_tipos_incidencias.png",
            mime="image/png"
        )
        
        # Tabla con detalles de distribución de tipos de incidencias (ahora desplegable)
        with st.expander("📋 Ver detalles de Distribución de Tipos de Incidencias"):
            distribucion_incidencias_df = df_filtrado['incidencia_tipo'].value_counts().reset_index()
            distribucion_incidencias_df.columns = ['Tipo de Incidencia', 'Número de Incidencias']
            distribucion_incidencias_df['Porcentaje (%)'] = (distribucion_incidencias_df['Número de Incidencias'] / df_filtrado.shape[0]) * 100
            st.table(distribucion_incidencias_df.style.format({'Porcentaje (%)': '{:.2f}%'}))   
    
        # Insight
        tipo_mas_frecuente = tipo_incidencias_porcentaje.idxmax()
        porcentaje_mas_frecuente = tipo_incidencias_porcentaje.max()
    
        insight_7 = f"""
        ### 💡 Insight 7: Tipo de Incidencia Más Común
        El tipo de incidencia más común es **{tipo_mas_frecuente}**, representando el **{porcentaje_mas_frecuente:.2f}%** de las incidencias totales. Esto sugiere que las **{tipo_mas_frecuente}s** son un factor significativo en el tráfico de la zona.
        """
        st.markdown(insight_7)
        insights.append(insight_7)


def crear_grafico_8(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 8: Top 10 Incidentes por Subtipo"""
    st.header("🏷️ 8. Top 10 Incidentes por Subtipo")
    
    # Contar el número de incidentes por subtipo en el DataFrame filtrado
    top_incidencias_subtipo = df_filtrado['incidencia_subtipo'].value_counts().head(10)
    
    if top_incidencias_subtipo.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        # Crear un DataFrame para facilitar la visualización
        top_incidencias_subtipo_df = top_incidencias_subtipo.reset_index()
        top_incidencias_subtipo_df.columns = ['Subtipo', 'Número de Incidencias']
    
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Número de Incidencias', y='Subtipo', data=top_incidencias_subtipo_df, palette="Blues_r")
        plt.title('🏷️ Top 10 Incidentes por Subtipo', fontsize=16)
        plt.xlabel('Número de Incidencias', fontsize=14)
        plt.ylabel('Subtipo', fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_top_incidencias_subtipo.png",
            mime="image/png"
        )
        
        # Tabla con detalles de Top 10 Incidentes por Subtipo (ahora desplegable)
        with st.expander("📋 Ver detalles de Top 10 Incidentes por Subtipo"):
            top_incidencias_subtipo_display = top_incidencias_subtipo.reset_index()
            top_incidencias_subtipo_display.columns = ['Subtipo', 'Número de Incidencias']
            top_incidencias_subtipo_display['Porcentaje (%)'] = (top_incidencias_subtipo_display['Número de Incidencias'] / df_filtrado.shape[0]) * 100
            st.table(top_incidencias_subtipo_display.style.format({'Porcentaje (%)': '{:.2f}%'}))   
        
        # Insight
        subtipo_mas_frecuente = top_incidencias_subtipo.index[0]
        num_incidencias_subtipo = top_incidencias_subtipo.iloc[0]
        porcentaje_subtipo_mas_frecuente = (num_incidencias_subtipo / df_filtrado.shape[0]) * 100
    
        insight_8 = f"""
        ### 💡 Insight 8: Subtipo de Incidencia Más Común
        El subtipo más frecuente es **{subtipo_mas_frecuente}**, con **{num_incidencias_subtipo} incidencias**, representando un **{porcentaje_subtipo_mas_frecuente:.2f}%** de las incidencias filtradas. Esto indica que este subtipo es especialmente relevante.
        """
        st.markdown(insight_8)
        insights.append(insight_8)


def crear_grafico_9(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 9: Porcentaje de Accidentes por Día Laboral vs. Fin de Semana"""
    st.header("📅 9. Porcentaje de Accidentes por Día Laboral vs. Fin de Semana")
    
    if df_filtrado_accidentes.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        # Crear una copia para evitar SettingWithCopyWarning
        df_filtrado_accidentes_copy = df_filtrado_accidentes.copy()
    
        # Crear una nueva columna que clasifique los días como "Laboral" o "Fin de Semana"
        df_filtrado_accidentes_copy['tipo_dia'] = df_filtrado_accidentes_copy['dia_semana'].apply(lambda x: 'Fin de Semana' if x >= 5 else 'Laboral')
    
        # Calcular el porcentaje de accidentes por tipo de día
        porcentaje_accidentes_por_tipo_dia = df_filtrado_accidentes_copy['tipo_dia'].value_counts(normalize=True) * 100
    
        # Crear un DataFrame para visualizar los resultados
        porcentaje_accidentes_df = porcentaje_accidentes_por_tipo_dia.reset_index()
        porcentaje_accidentes_df.columns = ['Tipo de Día', 'Porcentaje (%)']
    
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tipo de Día', y='Porcentaje (%)', data=porcentaje_accidentes_df, palette="Blues_r")
        plt.title('📅 Porcentaje de Accidentes por Día Laboral vs. Fin de Semana', fontsize=16)
        plt.xlabel('Tipo de Día', fontsize=14)
        plt.ylabel('Porcentaje (%)', fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_porcentaje_accidentes_tipo_dia.png",
            mime="image/png"
        )
    
        # Mostrar el DataFrame con los porcentajes (ahora desplegable)
        with st.expander("📋 Ver detalles de Porcentaje de Accidentes por Tipo de Día"):
            st.table(porcentaje_accidentes_df.style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        tipo_dia_mas_accidentes = porcentaje_accidentes_df.loc[porcentaje_accidentes_df['Porcentaje (%)'].idxmax(), 'Tipo de Día']
        porcentaje_mayor = porcentaje_accidentes_df['Porcentaje (%)'].max()
        num_accidentes_tipo_dia = int((porcentaje_mayor / 100) * total_accidentes_filtrado)
        # Continuación del Insight 9
        insight_9 = f"""
        ### 💡 Insight 9: Día con Más Accidentes
        Los accidentes ocurren más en **{tipo_dia_mas_accidentes}**, con **{num_accidentes_tipo_dia} accidentes**, representando el **{porcentaje_mayor:.2f}%** del total de accidentes. Esto sugiere que los **{tipo_dia_mas_accidentes.lower()}s** tienen una mayor siniestralidad.
        """
        st.markdown(insight_9)
        insights.append(insight_9)


def crear_grafico_10(df_filtrado_accidentes, total_accidentes_filtrado):
    """Gráfico 10: Número de Accidentes por Año"""
    st.header("📅 10. Número de Accidentes por Año")
    accidentes_por_annio = df_filtrado_accidentes['annio'].value_counts().sort_index()
    
    if accidentes_por_annio.empty:
        st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    else:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=accidentes_por_annio.index, y=accidentes_por_annio.values, palette="Blues_r")
        plt.title("📅 Número de Accidentes por Año")
        plt.xlabel("Año")
        plt.ylabel("Número de Accidentes")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Botón de descarga del gráfico
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="💾 Descargar gráfico",
            data=buf,
            file_name="grafico_accidentes_por_annio.png",
            mime="image/png"
        )
        
        # Tabla con detalles de accidentes por año (ahora desplegable)
        with st.expander("📋 Ver detalles de Accidentes por Año"):
            accidentes_por_annio_df = accidentes_por_annio.reset_index()
            accidentes_por_annio_df.columns = ['Año', 'Número de Accidentes']
            accidentes_por_annio_df['Porcentaje (%)'] = (accidentes_por_annio_df['Número de Accidentes'] / total_accidentes_filtrado) * 100
            st.table(accidentes_por_annio_df.style.format({'Porcentaje (%)': '{:.2f}%'}))
        
        # Insight
        año_pico = accidentes_por_annio_df.loc[accidentes_por_annio_df['Número de Accidentes'].idxmax(), 'Año']
        num_accidentes_año = accidentes_por_annio_df['Número de Accidentes'].max()
        porcentaje_año_pico = (num_accidentes_año / total_accidentes_filtrado) * 100
        
        insight_10 = f"""
        ### 💡 Insight 10: Año con Más Accidentes
        El año con más accidentes es **{año_pico}**, con **{num_accidentes_año} accidentes**, representando un **{porcentaje_año_pico:.2f}%** del total de accidentes en el rango seleccionado. Esto podría indicar tendencias o factores específicos de ese año.
        """
        st.markdown(insight_10)
        insights.append(insight_10)

# --------------------------- #
#     10. Mapa de Carreteras    #
# --------------------------- #

def crear_mapa_carreteras(carreteras_seleccionadas):
    """Mapa de Carreteras Seleccionadas"""
    st.title('📍 Mapa de Carreteras Seleccionadas')
    
    # Cargar puntos kilométricos desde un archivo GeoJSON (si lo tienes)
    try:
        gdf_puntos = gpd.read_file('puntos-kilometricos.geojson')
    
        # Filtrar los puntos kilométricos por las carreteras seleccionadas
        gdf_puntos_filtrados = gdf_puntos[gdf_puntos['via_nombre'].isin(carreteras_seleccionadas)]
    
        # Crear el mapa
        if not gdf_puntos_filtrados.empty:
            map_center = [gdf_puntos_filtrados['pk_latitud'].mean(), gdf_puntos_filtrados['pk_longitud'].mean()]
            mapa = folium.Map(location=map_center, zoom_start=10)
    
            # Añadir las líneas conectando los puntos de cada carretera
            for via, group in gdf_puntos_filtrados.groupby('via_nombre'):
                puntos = list(zip(group['pk_latitud'], group['pk_longitud']))
                folium.PolyLine(puntos, color='blue', weight=2.5, opacity=0.7).add_to(mapa)
    
            # Mostrar el mapa interactivo
            folium_static(mapa)
        else:
            st.warning("⚠️ No se encontraron puntos kilométricos para las carreteras seleccionadas.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el mapa debido a: {e}")

# --------------------------- #
#     11. Ejecución de Gráficos #
# --------------------------- #

# Mostrar el mapa de carreteras seleccionadas
crear_mapa_carreteras(carreteras_seleccionadas)

# Total de accidentes filtrados
total_accidentes_filtrado = df_filtrado_accidentes.shape[0]

# Crear y mostrar cada gráfico
crear_grafico_1(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_2(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_3(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_4(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_5(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_6(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_7(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_8(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_9(df_filtrado_accidentes, total_accidentes_filtrado)
crear_grafico_10(df_filtrado_accidentes, total_accidentes_filtrado)

# --------------------------- #
#   12. Modelado y Predicción  #
# --------------------------- #


# Definir los días de la semana manualmente en español
dias_semana_esp = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']

# Preprocesamiento de datos para el modelo
def preprocess_data(df):
    df_model = df.copy()
    df_model['hora'] = df_model['incidencia_fecha_inicio'].dt.hour
    df_model['dia_semana'] = df_model['incidencia_fecha_inicio'].dt.dayofweek
    df_model['es_accidente'] = df_model['es_accidente'].apply(lambda x: 1 if x == 'Accidente' else 0)

    # Codificar variables categóricas
    le_carretera = LabelEncoder()
    le_tramo = LabelEncoder()
    df_model['carretera_nombre_encoded'] = le_carretera.fit_transform(df_model['carretera_nombre'])
    df_model['tramo_nombre_encoded'] = le_tramo.fit_transform(df_model['tramo_nombre'])
    
    # Retornar los objetos necesarios
    return df_model, le_carretera, le_tramo

# Llamada a la función para preprocesar los datos y obtener las variables codificadas
df_model, le_carretera, le_tramo = preprocess_data(df)




# Seleccionar características y etiqueta
X = df_model[['carretera_nombre_encoded', 'tramo_nombre_encoded', 'hora', 'dia_semana']]
y = df_model['es_accidente']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Predicción de accidente según los filtros seleccionados por el usuario
st.header("🔮 Predicción de Accidente")

# Obtener la hora y el día actual con minutos en formato español
hora_actual = datetime.now().strftime('%H:%M')
dia_semana_actual = dias_semana_esp[datetime.now().weekday()]  # Obtener el día en español de la lista

# Filtros para la predicción
carretera_seleccionada = st.selectbox("🛣️ Selecciona una Carretera", df['carretera_nombre'].unique())
tramos_disponibles = ['Seleccionar Todos'] + df[df['carretera_nombre'] == carretera_seleccionada]['tramo_nombre'].unique().tolist()
tramo_seleccionado = st.selectbox("📍 Selecciona un Tramo", tramos_disponibles)


# Verificar si se seleccionó "Seleccionar Todos" para tramos
if tramo_seleccionado == 'Seleccionar Todos':
    tramos_seleccionados = df[df['carretera_nombre'] == carretera_seleccionada]['tramo_nombre'].unique().tolist()
    tramos_encoded = le_tramo.transform(tramos_seleccionados)  # Codificamos todos los tramos
else:
    tramos_encoded = [le_tramo.transform([tramo_seleccionado])[0]]  # Codificamos solo el tramo seleccionado

# Codificar la carretera seleccionada
carreteras_encoded = le_carretera.transform([carretera_seleccionada])[0]

# Crear el dataframe de entrada con las variables actuales
input_data = pd.DataFrame({
    'carretera_nombre_encoded': np.repeat(carreteras_encoded, len(tramos_encoded)),
    'tramo_nombre_encoded': tramos_encoded,
    'hora': np.repeat(datetime.now().hour, len(tramos_encoded)),
    'dia_semana': np.repeat(datetime.now().weekday(), len(tramos_encoded)),
    
})

# Hacer la predicción
pred_probs = model.predict_proba(input_data)[:, 1]  # Probabilidad de accidente

# Calcular la probabilidad promedio de accidente
avg_pred_prob = pred_probs.mean()

# Mostrar resultados en la app
st.write(f"🔮 **Probabilidad promedio de que ocurra un accidente**: {avg_pred_prob:.2%}")

# Explicación de los cálculos
st.markdown("""
**🔍 Explicación de los Cálculos**: Este valor es la probabilidad calculada de accidente usando los datos actuales (hora, día de la semana, tramo y tráfico). Refleja el riesgo general, pero no implica que siempre ocurra un accidente.
""")

# Explicación de la precisión del modelo
st.markdown(f"""
**🎯 Precisión del Modelo**:
La precisión muestra qué tan bien el modelo está funcionando en términos de clasificar correctamente los accidentes. Un valor de **{accuracy:.2f}** indica que el {accuracy*100:.0f}% de las veces, el modelo predijo correctamente si hubo o no un accidente.
""")
# Explicación precision modelo 2
st.markdown(f"""
**🎯 Explicación de la precisión del modelo**: 
Es importante tener en cuenta que la precisión de las predicciones del modelo puede verse afectada en ciertos tramos o carreteras debido a la falta de datos históricos suficientes. En estos casos, el modelo puede no tener la información necesaria para identificar patrones de accidentes y, como resultado, puede proporcionar probabilidades de accidente que sean muy bajas o incluso del **0%**.
""")

# Barómetro de Predicción utilizando Plotly
def crear_barometro(avg_pred_prob, hora_actual, dia_semana_actual):
    """Barómetro de Predicción utilizando Plotly"""
    st.header("📈 Barómetro de Predicción")

    # Mostrar la hora y el día actual encima del barómetro
    st.subheader(f"🕒 Hora actual: {hora_actual} | 📅 Día: {dia_semana_actual}")

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_pred_prob * 100,  # Convertir a porcentaje
        title = {'text': "Probabilidad de Accidente"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 33], 'color': "green"},
                {'range': [33, 66], 'color': "orange"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_pred_prob * 100
            }
        }
    ))
    
    # Mostrar la figura en la aplicación Streamlit
    st.plotly_chart(fig)
    
    # Explicación adicional
    st.markdown("""
    **🔍 Explicación del Barómetro:**
    - **Verde (0-33%)**: Baja probabilidad de accidente.
    - **Naranja (33-66%)**: Probabilidad moderada de accidente.
    - **Rojo (66-100%)**: Alta probabilidad de accidente.
    
    Este barómetro refleja el riesgo general basado en los filtros seleccionados. Un valor más alto indica una mayor probabilidad de que ocurra un accidente bajo las condiciones actuales.
    """)

# Llamar a la función para crear el barómetro
crear_barometro(avg_pred_prob, hora_actual, dia_semana_actual)

#---
# Obtener las importancias de las características
importancias = model.feature_importances_

# Nombres de las características
caracteristicas = ['carretera_nombre_encoded', 'tramo_nombre_encoded', 'hora', 'dia_semana']

# Crear un DataFrame con las importancias, multiplicándolas por 100 para convertirlas a porcentajes
df_importancias = pd.DataFrame({
    'Característica': caracteristicas,
    'Importancia (%)': importancias * 100  # Convertir a porcentaje
})

# Ordenar por importancia
df_importancias = df_importancias.sort_values(by='Importancia (%)', ascending=False)

# Mostrar la tabla de importancias en Streamlit
st.subheader("Importancia de las Características en el Modelo (%)")
st.dataframe(df_importancias)
