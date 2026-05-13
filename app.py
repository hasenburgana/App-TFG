import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Configuración de la página
st.set_page_config(page_title="TFG Perfiladoras", layout="wide")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Intentamos cargar el CSV
    try:
        df = pd.read_csv("data/perfiles_finales.csv", sep=",", decimal=".")
        # Limpieza básica de nombres de columnas
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No se encontraron datos en data/perfiles_finales.csv")
    st.stop()

# --- LÓGICA DE NEGOCIO ---
# Identificar columnas que son métricas (numéricas y no son IDs/Clusters)
ignored_cols = ['player_id', 'id', 'jugadora_id', 'player_name', 'name', 'jugadora', 
                'team_name', 'team', 'equipo', 'grupo_pos_clustering', 'grupo_pos', 
                'position', 'posicion', 'cluster_final', 'cluster']
metrics_cols = [c for c in df.columns if c not in ignored_cols and pd.api.types.is_numeric_dtype(df[c])]

# --- INTERFAZ: BARRA LATERAL ---
st.sidebar.header("Filtros de Búsqueda")
posiciones = sorted(df['grupo_pos_clustering'].unique())
pos_selected = st.sidebar.selectbox("Selecciona Posición", posiciones)

search = st.sidebar.text_input("Buscar Jugadora o Equipo").lower()

# Filtrado dinámico
df_filtered = df[df['grupo_pos_clustering'] == pos_selected]
if search:
    df_filtered = df_filtered[
        df_filtered['player_name'].str.lower().str.contains(search) | 
        df_filtered['team_name'].str.lower().str.contains(search)
    ]

# --- VISTA PRINCIPAL: LISTADO ---
st.title("⚽ TFG: Perfiladoras de Jugadoras")

selected_player_name = st.selectbox("Selecciona una jugadora para ver el detalle", 
                                   [""] + df_filtered['player_name'].tolist())

if selected_player_name:
    player_data = df[df['player_name'] == selected_player_name].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Equipo", player_data['team_name'])
        st.metric("Posición", player_data['grupo_pos_clustering'])
        st.metric("Cluster", int(player_data['cluster_final']))

    with col2:
        st.subheader(f"Métricas de {selected_player_name}")
        # Calculamos percentiles para las métricas
        player_metrics = player_data[metrics_cols]
        # Mostrar las 10 métricas principales en un gráfico de barras
        st.bar_chart(player_metrics.sort_values(ascending=False).head(10))

    # --- JUGADORAS SIMILARES ---
    st.divider()
    st.subheader("Jugadoras Similares (Mismo Perfil)")
    
    # Cálculo de distancia euclidiana (lo que hacía tu Java)
    group_data = df[df['grupo_pos_clustering'] == pos_selected]
    features = group_data[metrics_cols].fillna(0)
    target_features = player_data[metrics_cols].values.reshape(1, -1)
    
    distances = euclidean_distances(features, target_features).flatten()
    group_data = group_data.copy()
    group_data['similitud'] = distances
    
    similares = group_data[group_data['player_name'] != selected_player_name].sort_values('similitud').head(5)
    st.table(similares[['player_name', 'team_name', 'similitud']])

else:
    st.write(f"Mostrando {len(df_filtered)} jugadoras en la posición {pos_selected}")
    st.dataframe(df_filtered[['player_name', 'team_name', 'cluster_final']])
