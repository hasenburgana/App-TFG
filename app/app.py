from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import urllib.request
import urllib.error
import base64
import hashlib
import re
import os
import unicodedata
import urllib.parse
import json
import base64

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

DATA_PATH = Path("data/perfiles_finales.csv")
POSITION_COL = "grupo_pos_clustering"
CLUSTER_COL = "cluster_final"
PLAYER_COL = "player_name"
TEAM_COL = "team_name"
ID_COL = "player_id"

METADATA_COLS = {
    ID_COL, PLAYER_COL, TEAM_COL, "competicion", "posicion_principal",
    "grupo_pos", POSITION_COL, CLUSTER_COL, "cluster", "cluster_fa",
    "cluster_pca", "cluster_hier", "es_medoide", "nombre_medoide_cluster",
}

POS_METRICS = {
    "CB": [
        "pct_pases", "pct_pases_largos", "pct_pases_prog",
        "pct_pases_bajo_presion", "pases_ult_tercio_p90", "carries_prog_p90",
        "ratio_intercepciones_vs_entradas", "duelos_ter_ganados_p90",
        "despejes_p90", "pct_aereos", "pct_duelos_total", "recuperaciones_p90",
        "acciones_defensivas_campo_rival_p90",
    ],
    "LAT": [
        "pct_toques_en_campo_rival", "centros_p90", "pases_al_area_p90",
        "ratio_centros_vs_pases_al_area", "deep_progressions_p90",
        "xa_real_p90", "carries_prog_p90", "pct_regates",
        "ratio_intercepciones_vs_entradas", "acciones_defensivas_p90",
        "duelos_ter_ganados_p90", "despejes_p90", "pct_duelos_total",
        "recuperaciones_p90",
    ],
    "MCD": [
        "distancia_media_pases", "pct_pases_prog", "pct_pases_bajo_presion",
        "pases_progresivos_p90", "pases_largos_p90",
        "ratio_intercepciones_vs_entradas", "intercepciones_p90",
        "recuperaciones_p90", "acciones_agresivas_p90", "presiones_p90",
    ],
    "MC": [
        "carries_prog_p90", "velocidad_cond_m_s", "pct_pases",
        "pases_completados_p90", "pct_pases_prog", "distancia_media_pases",
        "pases_bajo_presion_p90", "xa_real_p90", "presiones_p90",
        "intercepciones_p90", "recuperaciones_p90", "pct_duelos_total",
        "pases_al_area_p90", "through_balls_p90", "pct_toques_en_area",
        "tiros_p90",
    ],
    "EXT": [
        "regates_p90", "pct_regates", "carries_prog_p90",
        "ratio_centros_vs_pases_al_area", "centros_p90", "pct_pases",
        "pases_completados_p90", "pct_toques_en_area", "tiros_p90",
        "xg_por_tiro", "xa_real_p90", "pases_clave_p90",
    ],
    "DEL": [
        "ratio_tiros_vs_pases", "xg_por_tiro", "distancia_media_tiros",
        "pct_toques_en_area", "tiros_puerta_p90", "pases_completados_p90",
        "pct_pases", "pct_pases_prog", "pases_clave_p90", "xa_real_p90",
        "through_balls_p90", "recibidos_de_espaldas_p90", "presiones_p90",
        "aereos_ganados_p90",
    ],
}

DERIVED_METRIC_PREFIXES = ("pct_", "ratio_", "rel_", "obv_")
DERIVED_METRIC_SUFFIXES = ("_p90",)
DERIVED_METRIC_NAMES = {
    "xg_por_tiro", "distancia_media_pases", "distancia_media_tiros",
    "posicion_media_x", "posicion_media_y", "std_posicion_x",
    "std_posicion_y", "velocidad_cond_m_s", "challenge_ratio",
}

# Cluster colours (up to 8 clusters)
CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#F4A261", "#457B9D",
    "#8E44AD", "#27AE60", "#F39C12", "#1ABC9C",
]

# Position labels in Spanish
POS_LABELS = {
    "CB": "Central", "LAT": "Lateral", "MCD": "Mediocentro Defensivo",
    "MC": "Mediocampista", "EXT": "Extremo", "DEL": "Delantera",
}

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Scout Profiles Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# PLAYER PHOTO HELPERS
# ──────────────────────────────────────────────

# --- CONFIGURACIÓN DE RUTAS ---
ASSETS_DIR = Path("assets")
PLAYERS_DIR = ASSETS_DIR / "players"
PLAYERS_DIR.mkdir(parents=True, exist_ok=True) # Crea la carpeta si no existe

# Diccionario para nombres complicados (StatsBomb -> Wikipedia)
MANUAL_NAME_MAP = {
    "Maria Francesca Caldentey Oliver": "Mariona Caldentey",
    "María Pilar León Cebrián": "Mapi León",
    "Patricia Guijarro Gutiérrez": "Patri Guijarro",
    "Aitana Bonmati Conca": "Aitana Bonmatí",
    "Salma Celeste Paralluelo Ayingono": "Salma Paralluelo",
    "Alba Maria Redondo Ferrer": "Alba Redondo",
    "Ona Batlle Pascual": "Ona Batlle"
}

def clean_accents(text: str) -> str:
    """Elimina tildes y normaliza texto."""
    if not text: return ""
    text = unicodedata.normalize('NFD', text)
    return "".join(c for c in text if unicodedata.category(c) != 'Mn')

def get_safe_path(name: str) -> str:
    """Crea un nombre de archivo seguro: 'Aitana Bonmatí' -> 'aitana_bonmati'"""
    return clean_accents(name).strip().replace(" ", "_").lower()

def fetch_wiki_url(name: str) -> str | None:
    """Busca en la API de Wikipedia la URL de la imagen original."""
    # Usamos el mapa manual si existe, si no, el nombre tal cual
    search_name = MANUAL_NAME_MAP.get(name, name)
    # Reemplazamos espacios por guiones bajos para Wikipedia
    wiki_name = urllib.parse.quote(search_name.replace(" ", "_"))
    
    # Intentamos en español y luego en inglés
    for lang in ['es', 'en']:
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"
            req = urllib.request.Request(url, headers={'User-Agent': 'ScoutingApp/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if "originalimage" in data:
                    return data["originalimage"]["source"]
        except:
            continue
    return None

def get_player_assets(player_name: str, team_name: str):
    """
    Intenta cargar foto y escudo localmente. 
    Si no existen, los descarga de Wikipedia.
    """
    team_slug = get_safe_path(team_name)
    player_slug = get_safe_path(player_name)
    
    # Carpeta por equipo: assets/players/barcelona/
    team_dir = PLAYERS_DIR / team_slug
    team_dir.mkdir(parents=True, exist_ok=True)
    
    player_path = team_dir / f"{player_slug}.png"
    badge_path = team_dir / "escudo.png"

    # --- Lógica para la Jugadora ---
    if not player_path.exists():
        url = fetch_wiki_url(player_name)
        if url:
            try:
                urllib.request.urlretrieve(url, player_path)
            except: pass

    # --- Lógica para el Escudo ---
    if not badge_path.exists():
        url = fetch_wiki_url(team_name)
        if url:
            try:
                urllib.request.urlretrieve(url, badge_path)
            except: pass

    # Convertir a Base64 para mostrar en Streamlit
    def to_b64(p):
        if p.exists():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    return to_b64(player_path), to_b64(badge_path)


#########
def _player_cache_key(name: str) -> str:
    return hashlib.md5(name.lower().strip().encode()).hexdigest()


# ── HEADERS REALISTAS (Vital para que no nos bloqueen) ──
FOTMOB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.fotmob.com/",
    "Accept": "application/json, text/plain, */*",
}

######
def _search_player_fotmob_id(name: str) -> str | None:
    """Busca en FotMob con limpieza de tildes y diccionario manual."""
    
    search_name = MANUAL_NAME_MAP.get(name.strip(), name.strip())

    def do_search(query: str) -> str | None:
        try:
            query_clean = clean_accents(query)
            clean_url = urllib.parse.quote(query_clean)
            
            url = f"https://apigw.fotmob.com/searchapi/suggest?term={clean_url}&lang=es"
            req = urllib.request.Request(url, headers=FOTMOB_HEADERS)
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                hits = data.get("squadMember", []) + data.get("player", [])
                for hit in hits:
                    pid = hit.get("participantId") or hit.get("id")
                    if pid:
                        return str(pid)
        except Exception as e:
            print(f"⚠️ Error buscando en FotMob a '{query}': {e}")
            pass
        return None

    # Intento 1: Nombre tal cual (o el mapeado)
    pid = do_search(search_name)
    if pid: return pid

    # Intento 2: Combinaciones si el nombre es muy largo (ej. StatsBomb)
    parts = search_name.split()
    if len(parts) > 2:
        # Ejemplo: Maria (0) Francesca (1) Caldentey (-2) Oliver (-1)
        
        # Intento 2.1: Nombre + 1er Apellido (Maria Caldentey)
        pid = do_search(f"{parts[0]} {parts[-2]}")
        if pid: return pid
        
        # Intento 2.2: Nombre + 2do Apellido (Maria Oliver)
        pid = do_search(f"{parts[0]} {parts[-1]}")
        if pid: return pid
        
        # Intento 2.3: 2do Nombre + 1er Apellido (Francesca Caldentey)
        pid = do_search(f"{parts[1]} {parts[-2]}")
        if pid: return pid

    print(f"❌ FotMob ID no encontrado para: {name}")
    return None


def _fetch_fotmob_photo(player_id: str, size: int = 96) -> bytes | None:
    """Fetch player photo from FotMob CDN."""
    url = f"https://images.fotmob.com/image_resources/playerimages/{player_id}.png"
    try:
        req = urllib.request.Request(url, headers=FOTMOB_HEADERS)
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read()
            if len(data) > 2000:  # real image, not empty placeholder
                return data
    except Exception:
        pass
    return None
######

@st.cache_data(show_spinner=False, ttl=86400)
def get_player_photo_b64(player_name: str) -> str | None:
    """Return base64-encoded PNG of the player, cached to disk."""
    key = _player_cache_key(player_name)
    cache_file = PHOTO_CACHE_DIR / f"{key}.png"
    miss_file = PHOTO_CACHE_DIR / f"{key}.miss"

    # Already tried and failed
    if miss_file.exists():
        return None

    # Disk cache hit
    if cache_file.exists():
        return base64.b64encode(cache_file.read_bytes()).decode()

    # Try FotMob search → photo
    pid = _search_player_fotmob_id(player_name)
    if pid:
        photo_bytes = _fetch_fotmob_photo(pid)
        if photo_bytes:
            cache_file.write_bytes(photo_bytes)
            return base64.b64encode(photo_bytes).decode()

    # Mark as miss so we don't keep trying
    miss_file.touch()
    return None


# def player_avatar_html(player_name: str, team_name: str, size: int = 80, border_color: str = "#2A9D8F", fetch_photo: bool = True) -> str:
#     """Versión con team_name obligatorio para organización estricta."""
#     team_slug = get_safe_path(team_name)
#     player_slug = get_safe_path(player_name)
    
#     team_dir = Path("assets/players") / team_slug
#     team_dir.mkdir(parents=True, exist_ok=True)
    
#     player_path = team_dir / f"{player_slug}.png"
    
#     # Descarga si no existe
#     if fetch_photo and not player_path.exists():
#         url = fetch_wiki_url(player_name)
#         if url:
#             try:
#                 urllib.request.urlretrieve(url, player_path)
#             except: pass
    
#     # Cargar imagen en Base64
#     img_b64 = None
#     if player_path.exists():
#         with open(player_path, "rb") as f:
#             img_b64 = base64.b64encode(f.read()).decode()

#     if img_b64:
#         return f'<img src="data:image/png;base64,{img_b64}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;border:3px solid {border_color};">'
    
#     # Fallback si no hay foto: iniciales
#     parts = player_name.strip().split()
#     initials = "".join(p[0].upper() for p in parts[:2]) if parts else "?"
#     return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{border_color};display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:{size//3}px;border:3px solid {border_color};">{initials}</div>'
def player_avatar_html(
    player_name: str,
    team_name: str = None
    size: int = 72,
    border_color: str = "#2A9D8F",
    fetch_photo: bool = True
) -> str:
    """Retorna el HTML de la imagen buscando primero en el JSON local."""
    img_url = None
    
    # 1. Buscar en el JSON cargado
    if player_name in CUSTOM_PHOTOS:
        img_url = CUSTOM_PHOTOS[player_name]

    # 2. Si existe en el JSON, devolvemos el HTML con esa URL
    if img_url:
        return (
            f'<img src="{img_url}" '
            f'style="width:{size}px;height:{size}px;border-radius:50%;'
            f'object-fit:cover;border:2.5px solid {border_color};'
            f'background:#0d1b2a;" />'
        )

    # 3. Si NO está en el JSON, intentar la lógica original de FotMob (Base64)
    b64 = None
    if fetch_photo:
        try:
            # Aquí asumo que mantienes tu función original get_player_photo_b64
            b64 = get_player_photo_b64(player_name)
        except Exception:
            pass

    if b64:
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{size}px;height:{size}px;border-radius:50%;'
            f'object-fit:cover;border:2.5px solid {border_color};'
            f'background:#0d1b2a;" />'
        )

    # 4. Fallback: Iniciales si nada de lo anterior funciona
    parts = player_name.strip().split()
    initials = "".join(p[0].upper() for p in parts[:2]) if parts else "?"
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:{border_color};border:2.5px solid {border_color};'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:{size // 3}px;font-weight:800;color:#fff;'
        f'font-family:\'DM Sans\',sans-serif;">'
        f'{initials}</div>'
    )

# Need urllib.parse for quote
import urllib.parse


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────

def inject_css() -> None:
    css = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700;800&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] { background: #08111e !important; color: #e2e8f2; }
.stApp { background: #08111e !important; font-family: 'DM Sans', sans-serif; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d1b2a !important; border-right: 1px solid #1a2e45; }
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stTextInput label { color: #7a9bb5 !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: #0a1626 !important; border: 1px solid #1a2e45 !important; border-radius: 8px; padding: 10px 14px; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #2A9D8F !important; font-family: 'Space Mono', monospace !important; font-size: 1.4rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #7a9bb5 !important; }

/* ── Streamlit overrides ── */
.stSelectbox > div > div, .stTextInput > div > div > input { background: #0d1b2a !important; border: 1px solid #1a2e45 !important; color: #e2e8f2 !important; border-radius: 8px !important; }
.stRadio > div { gap: 6px; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #c8d8e8; }
.stMultiSelect > div { background: #0d1b2a !important; border: 1px solid #1a2e45 !important; color: #e2e8f2 !important; border-radius: 8px !important; }
.stSlider [data-testid="stSliderThumb"] { background: #2A9D8F !important; }
.stSlider [data-testid="stSliderTrack"] { background: #1a2e45 !important; }
div[data-testid="stMetric"] { background: #0d1b2a; border: 1px solid #1a2e45; border-radius: 10px; padding: 14px 18px; }
div[data-testid="stMetricValue"] { color: #2A9D8F !important; font-family: 'Space Mono', monospace; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #7a9bb5 !important; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.stDataFrame { background: #0d1b2a !important; border-radius: 10px; border: 1px solid #1a2e45; }
.stDataFrame th { background: #0a1626 !important; color: #7a9bb5 !important; font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; }
.stDataFrame td { color: #c8d8e8 !important; background: #0d1b2a !important; }
.stExpander { background: #0d1b2a !important; border: 1px solid #1a2e45 !important; border-radius: 10px !important; }
.stExpander summary { color: #c8d8e8 !important; }
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif; color: #e2e8f2 !important; }
.stCaption, .stCaption p { color: #4a6580 !important; }
.stInfo { background: #0d2235 !important; border: 1px solid #1a3a55 !important; color: #7ab8d8 !important; border-radius: 8px; }
.stWarning { background: #1c1500 !important; border: 1px solid #3d3000 !important; border-radius: 8px; }
div[data-testid="column"] { background: transparent !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1b2a; border-radius: 10px; gap: 4px; padding: 4px; border: 1px solid #1a2e45; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 7px; color: #7a9bb5; font-weight: 600; font-size: .88rem; }
.stTabs [aria-selected="true"] { background: #2A9D8F !important; color: #ffffff !important; }

/* ── Custom components ── */
.hero-wrap { background: linear-gradient(135deg, #0d2235 0%, #0a1626 60%, #091520 100%); border: 1px solid #1a3a55; border-radius: 14px; padding: 28px 32px 24px; margin-bottom: 22px; position: relative; overflow: hidden; }
.hero-wrap::before { content: ''; position: absolute; top: -60px; right: -60px; width: 220px; height: 220px; background: radial-gradient(circle, rgba(42,157,143,0.18) 0%, transparent 70%); pointer-events: none; }
.hero-kicker { color: #2A9D8F; text-transform: uppercase; font-weight: 800; font-size: .72rem; letter-spacing: .1em; margin-bottom: 10px; }
.hero-title { font-size: 2.4rem; line-height: 1.06; margin: 0 0 10px; font-weight: 800; color: #e2e8f2; }
.hero-title span { color: #2A9D8F; }
.hero-copy { color: #6a8aa5; font-size: .95rem; max-width: 680px; margin: 0; line-height: 1.6; }
.stat-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 18px 0 6px; }
.stat-chip { background: #0a1626; border: 1px solid #1a2e45; border-radius: 10px; padding: 12px 16px; }
.stat-chip .label { color: #4a6580; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; margin-bottom: 4px; }
.stat-chip .value { color: #2A9D8F; font-size: 1.5rem; font-family: 'Space Mono', monospace; font-weight: 700; line-height: 1; }

.player-header { background: linear-gradient(135deg, #0d2235, #091520); border: 1px solid #1a3a55; border-radius: 14px; padding: 20px 24px; margin-bottom: 18px; display: flex; align-items: center; gap: 20px; }
.player-info { flex: 1; }
.player-name { font-size: 1.9rem; font-weight: 800; color: #e2e8f2; margin: 0 0 4px; line-height: 1.1; }
.player-meta { color: #4a6580; font-size: .88rem; margin-bottom: 10px; }
.cluster-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: .78rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: #fff; }
.pos-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: .75rem; font-weight: 700; background: #1a2e45; color: #7ab8d8; margin-right: 6px; }

.section-title { font-size: 1.05rem; font-weight: 800; color: #e2e8f2; text-transform: uppercase; letter-spacing: .06em; padding-bottom: 8px; border-bottom: 1px solid #1a2e45; margin-bottom: 14px; }
.section-title span { color: #2A9D8F; margin-right: 6px; }

.card { background: #0d1b2a; border: 1px solid #1a2e45; border-radius: 12px; padding: 18px 20px; }
.card-sm { background: #0d1b2a; border: 1px solid #1a2e45; border-radius: 10px; padding: 12px 14px; }

.interp-box { background: #071220; border-left: 3px solid #2A9D8F; border-radius: 0 8px 8px 0; padding: 12px 16px; color: #7ab8d8; font-size: .9rem; line-height: 1.6; margin-bottom: 12px; }

.metric-pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px; border-radius: 20px; font-size: .8rem; font-weight: 700; margin: 3px; }
.pill-strength { background: rgba(42,157,143,0.15); border: 1px solid rgba(42,157,143,0.4); color: #2A9D8F; }
.pill-weakness { background: rgba(230,57,70,0.12); border: 1px solid rgba(230,57,70,0.35); color: #E63946; }
.z-value { font-family: 'Space Mono', monospace; font-size: .75rem; opacity: .8; }

.sim-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; margin-top: 4px; }
.sim-card { background: #071220; border: 1px solid #1a2e45; border-radius: 12px; padding: 14px 12px; text-align: center; transition: border-color .2s, transform .15s; }
.sim-card:hover { border-color: #2A9D8F; transform: translateY(-2px); }
.sim-card .sim-name { font-weight: 700; font-size: .82rem; color: #c8d8e8; margin: 8px 0 3px; line-height: 1.2; }
.sim-card .sim-team { font-size: .72rem; color: #4a6580; margin-bottom: 5px; }
.sim-card .sim-dist { font-family: 'Space Mono', monospace; font-size: .72rem; color: #2A9D8F; }
.sim-card .sim-cluster { display: inline-block; margin-top: 4px; padding: 2px 8px; border-radius: 20px; font-size: .68rem; font-weight: 700; color: #fff; }

.sidebar-brand { background: linear-gradient(135deg, #0a1e33, #071220); border: 1px solid #1a3a55; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }
.sidebar-brand strong { display: block; font-size: 1.1rem; color: #e2e8f2 !important; margin-bottom: 3px; }
.sidebar-brand .sub { color: #4a6580 !important; font-size: .8rem; }
.sidebar-brand .accent { color: #2A9D8F !important; }

.glossary-term { font-weight: 800; color: #2A9D8F; }
.glossary-line { color: #6a8aa5; font-size: .88rem; margin-bottom: .65rem; line-height: 1.55; }

@media (max-width: 900px) { .hero-title { font-size: 1.7rem; } .stat-strip { grid-template-columns: 1fr; } .sim-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
    """
    try:
        st.html(css)
    except AttributeError:
        st.markdown(css, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
# --- Carga de fotos personalizadas ---
@st.cache_data
def load_custom_photos():
    try:
        # Importante: verifica que el nombre del archivo sea exacto
        with open("jugadoras_fotos.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar jugadoras_fotos.json: {e}")
        return {}

CUSTOM_PHOTOS = load_custom_photos()

@st.cache_data(show_spinner=False)
def read_csv_from_bytes(file_mtime: float | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        st.error(
            "No encuentro `data/perfiles_finales.csv`. Coloca el CSV exportado "
            "desde Colab en esa ruta dentro del repositorio."
        )
        return None
    file_mtime = DATA_PATH.stat().st_mtime
    df = read_csv_from_bytes(file_mtime)
    df = normalise_required_columns(df)
    return df


def normalise_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if POSITION_COL not in df.columns and "grupo_pos" in df.columns:
        df[POSITION_COL] = df["grupo_pos"]
    if CLUSTER_COL not in df.columns:
        for candidate in ["cluster", "cluster_fa", "cluster_pca", "cluster_hier"]:
            if candidate in df.columns:
                df[CLUSTER_COL] = df[candidate]
                break
    if TEAM_COL not in df.columns:
        df[TEAM_COL] = "Sin equipo"
    if ID_COL not in df.columns:
        df[ID_COL] = np.arange(len(df)).astype(str)

    missing = [col for col in [PLAYER_COL, POSITION_COL, CLUSTER_COL] if col not in df.columns]
    if missing:
        st.error(f"Faltan columnas necesarias en el CSV: {', '.join(missing)}")
        st.stop()

    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce").fillna(-1).astype(int)
    df[PLAYER_COL] = df[PLAYER_COL].astype(str)
    df[TEAM_COL] = df[TEAM_COL].fillna("Sin equipo").astype(str)
    df[POSITION_COL] = df[POSITION_COL].astype(str)
    return df


# ──────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────

def metric_columns(df: pd.DataFrame, position: str | None = None) -> list[str]:
    source = df[df[POSITION_COL] == position] if position else df
    numeric_cols = source.select_dtypes(include=np.number).columns.tolist()
    if position in POS_METRICS:
        return [m for m in POS_METRICS[position] if m in numeric_cols]
    return sorted(
        col for col in numeric_cols
        if col not in METADATA_COLS and not col.startswith("cluster")
        and (
            col.endswith(DERIVED_METRIC_SUFFIXES)
            or col.startswith(DERIVED_METRIC_PREFIXES)
            or col in DERIVED_METRIC_NAMES
        )
    )


def all_derived_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return sorted(
        col for col in numeric_cols
        if col not in METADATA_COLS and not col.startswith("cluster")
        and (
            col.endswith(DERIVED_METRIC_SUFFIXES)
            or col.startswith(DERIVED_METRIC_PREFIXES)
            or col in DERIVED_METRIC_NAMES
        )
    )


def clean_metric_name(metric: str) -> str:
    return metric.replace("_p90", "").replace("pct_", "% ").replace("_", " ").title()


def metric_family(metric: str) -> str:
    m = metric.lower()
    if any(t in m for t in ["xg", "tiros", "goles", "area", "espaldas"]):
        return "finalización y presencia en área"
    if any(t in m for t in ["pases", "through", "xa", "asistencias", "centros"]):
        return "pase y creación"
    if any(t in m for t in ["carries", "regates", "progressions", "condu", "velocidad"]):
        return "conducción y progresión"
    if any(t in m for t in ["presiones", "recuperaciones", "intercepciones", "defensivas", "despejes", "duelos", "aereos"]):
        return "defensa y recuperación"
    if any(t in m for t in ["posicion", "std_"]):
        return "posicionamiento"
    return "perfil mixto"


def percentile(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return 0.0
    return float((clean <= value).mean() * 100)


def top_discriminant_metrics(df_pos: pd.DataFrame, metrics: Iterable[str], limit: int = 12) -> list[str]:
    scores: list[tuple[str, float]] = []
    for metric in metrics:
        means = df_pos.groupby(CLUSTER_COL)[metric].mean(numeric_only=True)
        if len(means) > 1:
            scores.append((metric, float(means.max() - means.min())))
    return [m for m, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:limit]]


def scaled_p05_p95_values(df_pos: pd.DataFrame, row: pd.Series, metrics: list[str]) -> dict[str, float]:
    p05 = df_pos[metrics].quantile(0.05)
    p95 = df_pos[metrics].quantile(0.95)
    diff = (p95 - p05).replace(0, 1)
    values = ((row[metrics] - p05) / diff).clip(0, 1) * 0.70 + 0.15
    return values.astype(float).to_dict()


def cluster_mean_row(df_pos: pd.DataFrame, cluster: int, metrics: list[str]) -> pd.Series:
    return df_pos[df_pos[CLUSTER_COL] == cluster][metrics].mean(numeric_only=True)


def cluster_percentiles(df_pos: pd.DataFrame, cluster: int, metrics: Iterable[str]) -> dict[str, float]:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    return {
        metric: percentile(df_pos[metric], df_cluster[metric].mean())
        for metric in metrics
    }


def cluster_medoid_player(df_pos: pd.DataFrame, cluster: int) -> pd.Series:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    if df_cluster.empty:
        return df_pos.iloc[0]
    if "es_medoide" in df_cluster.columns:
        mask = df_cluster["es_medoide"].astype(str).str.lower().isin(["true", "1", "si", "sí"])
        if mask.any():
            return df_cluster[mask].iloc[0]
    if "nombre_medoide_cluster" in df_cluster.columns:
        medoid_name = str(df_cluster["nombre_medoide_cluster"].dropna().iloc[0]).strip()
        if medoid_name:
            same_name = df_pos[df_pos[PLAYER_COL].astype(str).str.strip() == medoid_name]
            same_cluster = same_name[same_name[CLUSTER_COL] == cluster]
            if not same_cluster.empty:
                return same_cluster.iloc[0]
            if not same_name.empty:
                return same_name.iloc[0]
    return df_cluster.iloc[0]


def radar_angles(n: int) -> list[float]:
    return [i * 360.0 / n for i in range(n)]


def similar_players(df_pos: pd.DataFrame, player: pd.Series, metrics: list[str], limit: int = 10) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    values = df_pos[metrics].apply(pd.to_numeric, errors="coerce").fillna(df_pos[metrics].median())
    std = values.std(ddof=0).replace(0, 1)
    z_values = (values - values.mean()) / std
    idx = player.name
    distances = np.sqrt(((z_values - z_values.loc[idx]) ** 2).sum(axis=1))
    result = df_pos.copy()
    result["distancia"] = distances
    result = result[result.index != idx]
    return result.sort_values("distancia").head(limit)


def cluster_z_summary(df_pos: pd.DataFrame, cluster: int, metrics: Iterable[str]) -> pd.DataFrame:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    rows = []
    for metric in metrics:
        g_mean = df_pos[metric].mean()
        g_std = df_pos[metric].std(ddof=0)
        c_mean = df_cluster[metric].mean()
        z = 0 if g_std == 0 or pd.isna(g_std) else (c_mean - g_mean) / g_std
        rows.append({"metric": metric, "z": z})
    return pd.DataFrame(rows).sort_values("z", ascending=False)


def automatic_cluster_interpretation(df_pos: pd.DataFrame, cluster: int, metrics: list[str]) -> dict:
    summary = cluster_z_summary(df_pos, cluster, metrics)
    highs = summary.head(4)
    lows = summary.tail(3).sort_values("z")
    cluster_size = int((df_pos[CLUSTER_COL] == cluster).sum())
    families = highs["metric"].map(metric_family).value_counts()
    dominant_family = families.index[0] if not families.empty else "perfil mixto"
    high_names = [clean_metric_name(m) for m in highs["metric"].tolist()]
    low_names = [clean_metric_name(m) for m in lows["metric"].tolist()]
    text = (
        f"Cluster {cluster}: perfil orientado a {dominant_family}. "
        f"Destaca por {', '.join(high_names[:3])}. "
        f"Sus valores relativamente más bajos aparecen en {', '.join(low_names[:2])}. "
        f"Contiene {cluster_size} jugadoras."
    )
    return {
        "cluster": cluster, "familia": dominant_family,
        "interpretacion": text, "fortalezas": ", ".join(high_names),
        "debilidades": ", ".join(low_names), "n": cluster_size,
    }


def all_cluster_interpretations(df_pos: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = [
        automatic_cluster_interpretation(df_pos, int(c), metrics)
        for c in sorted(df_pos[CLUSTER_COL].dropna().unique())
    ]
    return pd.DataFrame(rows)


def representative_players(df_pos: pd.DataFrame, metrics: list[str]) -> dict[int, str]:
    if not metrics:
        return {}
    values = df_pos[metrics].apply(pd.to_numeric, errors="coerce").fillna(df_pos[metrics].median())
    std = values.std(ddof=0).replace(0, 1)
    z_values = (values - values.mean()) / std
    result = {}
    for cluster, cdf in df_pos.groupby(CLUSTER_COL):
        cz = z_values.loc[cdf.index]
        centroid = cz.mean()
        dists = np.sqrt(((cz - centroid) ** 2).sum(axis=1))
        result[int(cluster)] = str(df_pos.loc[dists.idxmin(), PLAYER_COL])
    return result


def profile_score(df_pos: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    active = {m: w for m, w in weights.items() if w != 0}
    if not active:
        return pd.DataFrame()
    scores = pd.Series(0.0, index=df_pos.index)
    total_weight = 0.0
    for metric, raw_weight in active.items():
        direction = 1 if raw_weight > 0 else -1
        weight = abs(raw_weight)
        ranks = df_pos[metric].rank(pct=True).fillna(0.0)
        component = ranks if direction > 0 else 1 - ranks
        scores += component * weight
        total_weight += weight
    result = df_pos[[PLAYER_COL, TEAM_COL, POSITION_COL, CLUSTER_COL]].copy()
    result["score"] = (scores / total_weight * 100).round(2)
    return result.sort_values("score", ascending=False)


# ──────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c8d8e8"),
)


def _polar_layout(height: int = 520) -> dict:
    return dict(
        height=height,
        margin=dict(l=60, r=60, t=60, b=50),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.15, 0.5, 0.85],
                ticktext=["bajo", "medio", "alto"],
                gridcolor="#1a2e45", tickfont=dict(size=9, color="#4a6580"),
                linecolor="#1a2e45",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#7a9bb5"),
                gridcolor="#1a2e45", linecolor="#1a2e45",
                rotation=90, direction="clockwise",
            ),
        ),
        legend=dict(orientation="h", y=-0.1, font=dict(size=11, color="#c8d8e8")),
        **_PLOTLY_LAYOUT,
    )


def player_cluster_radar_figure(
    player_name: str, cluster: int,
    player_vals: dict[str, float], rep_name: str, rep_vals: dict[str, float],
    position: str,
) -> go.Figure:
    metrics = list(player_vals.keys())
    labels = [clean_metric_name(m) for m in metrics]
    angles = radar_angles(len(metrics))
    closed = angles + [360.0]
    pv = [player_vals[m] for m in metrics]
    rv = [rep_vals[m] for m in metrics]
    cluster_color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pv + pv[:1], theta=closed, mode="lines+markers", fill="toself",
        name=player_name[:26],
        line=dict(color=cluster_color, width=3),
        marker=dict(size=6),
        text=labels + labels[:1],
        hovertemplate="%{text}<br>Escala: %{r:.2f}<extra></extra>",
        fillcolor=f"rgba{(*_hex_to_rgb(cluster_color), 0.18)}",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rv + rv[:1], theta=closed, mode="lines+markers", fill="toself",
        name=f"Perfil típico C{cluster} ({rep_name[:20]})",
        line=dict(color="#4a6580", width=2, dash="dot"),
        marker=dict(size=4),
        text=labels + labels[:1],
        hovertemplate="%{text}<br>Escala: %{r:.2f}<extra></extra>",
        fillcolor="rgba(74,101,128,0.08)",
    ))
    layout = _polar_layout(600)
    layout["polar"]["angularaxis"]["tickvals"] = angles
    layout["polar"]["angularaxis"]["ticktext"] = labels
    layout["polar"]["angularaxis"]["tickmode"] = "array"
    layout["title"] = dict(text=f"{player_name} · {position}", font=dict(size=15, color="#e2e8f2"), x=0)
    fig.update_layout(**layout)
    return fig


def cluster_radar_figure(df_pos: pd.DataFrame, metrics: list[str], position: str, top_n: int = 15) -> go.Figure:
    medias = df_pos.groupby(CLUSTER_COL)[metrics].mean(numeric_only=True).sort_index()
    sel = top_discriminant_metrics(df_pos, metrics, top_n) or metrics[:top_n]
    norm = pd.DataFrame(index=medias.index, columns=sel, dtype=float)
    for metric in sel:
        norm[metric] = [percentile(df_pos[metric], medias.loc[c, metric]) for c in medias.index]

    rep_names = representative_players(df_pos, sel)
    labels = [clean_metric_name(m) for m in sel]
    n_clusters = len(norm)
    cols = min(3, n_clusters)
    rows_n = int(np.ceil(n_clusters / cols))
    specs = [[{"type": "polar"} for _ in range(cols)] for _ in range(rows_n)]
    titles = [
        f"Cluster {c}<br><sup>{rep_names.get(c, '—')}</sup>"
        for c in norm.index
    ]
    fig = make_subplots(rows=rows_n, cols=cols, specs=specs, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.16)
    for i, cluster in enumerate(norm.index):
        row, col = i // cols + 1, i % cols + 1
        values = norm.loc[cluster].astype(float).tolist()
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=labels + labels[:1],
            fill="toself", name=f"Cluster {cluster}",
            line=dict(color=color, width=2),
            fillcolor=f"rgba{(*_hex_to_rgb(color), 0.45)}",
            hovertemplate="%{theta}<br>Percentil %{r:.0f}%<extra></extra>",
        ), row=row, col=col)
        pname = "polar" if i == 0 else f"polar{i + 1}"
        fig.layout[pname].update(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75],
                            ticktext=["", "50%", ""], showline=False,
                            gridcolor="#1a2e45", tickfont=dict(size=8, color="#4a6580")),
            angularaxis=dict(tickfont=dict(size=9, color="#7a9bb5"), gridcolor="#1a2e45"),
        )
    fig.update_layout(
        title=dict(text=f"Perfiles tácticos por percentil · {position}",
                   font=dict(size=15, color="#e2e8f2"), x=0),
        height=380 * rows_n + 90, showlegend=False,
        margin=dict(l=30, r=30, t=90, b=35),
        **_PLOTLY_LAYOUT,
    )
    return fig


def player_cluster_bar_figure(
    player_name: str, cluster: int,
    player_pct: dict[str, float], cluster_pct: dict[str, float],
) -> go.Figure:
    metrics = list(player_pct.keys())
    labels = [clean_metric_name(m) for m in metrics]
    cluster_color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[player_pct[m] for m in metrics],
        name=player_name[:22], orientation="h",
        marker_color=cluster_color,
        hovertemplate="%{y}<br>Percentil %{x:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[cluster_pct[m] for m in metrics],
        name=f"Media cluster {cluster}", orientation="h",
        marker_color="#4a6580",
        hovertemplate="%{y}<br>Percentil %{x:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        height=max(420, 34 * len(metrics) + 120),
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#1a2e45", tickfont=dict(color="#7a9bb5")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#c8d8e8")),
        legend=dict(orientation="h", font=dict(color="#c8d8e8")),
        margin=dict(l=130, r=25, t=25, b=45),
        **_PLOTLY_LAYOUT,
    )
    return fig


def profile_radar_figure(df_pos: pd.DataFrame, weights: dict[str, int], scores: pd.DataFrame) -> go.Figure:
    active = {m: w for m, w in weights.items() if w != 0}
    metrics = list(active.keys())
    labels = [clean_metric_name(m) for m in metrics]
    angles = radar_angles(len(metrics))
    closed = angles + [360.0]
    desired = [0.5 + np.sign(active[m]) * abs(active[m]) / 5.0 * 0.35 for m in metrics]

    best_vals, best_name = [], "Mejor encaje"
    if not scores.empty:
        br = scores.iloc[0]
        best_name = str(br[PLAYER_COL])
        bm = df_pos[
            (df_pos[PLAYER_COL] == br[PLAYER_COL]) & (df_pos[TEAM_COL] == br[TEAM_COL])
        ]
        if bm.empty:
            bm = df_pos[df_pos[PLAYER_COL] == br[PLAYER_COL]]
        if not bm.empty:
            bvd = scaled_p05_p95_values(df_pos, bm.iloc[0], metrics)
            best_vals = [bvd[m] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=desired + desired[:1], theta=closed, mode="lines+markers", fill="toself",
        name="Perfil buscado", line=dict(color="#2A9D8F", width=3),
        marker=dict(size=6), text=labels + labels[:1],
        hovertemplate="%{text}<br>Objetivo: %{r:.2f}<extra></extra>",
        fillcolor="rgba(42,157,143,0.12)",
    ))
    if best_vals:
        fig.add_trace(go.Scatterpolar(
            r=best_vals + best_vals[:1], theta=closed, mode="lines+markers", fill="toself",
            name=f"Mejor encaje: {best_name[:24]}",
            line=dict(color="#E63946", width=2.5), marker=dict(size=5),
            text=labels + labels[:1],
            hovertemplate="%{text}<br>Escala: %{r:.2f}<extra></extra>",
            fillcolor="rgba(230,57,70,0.10)",
        ))
    layout = _polar_layout(520)
    layout["polar"]["angularaxis"]["tickvals"] = angles
    layout["polar"]["angularaxis"]["ticktext"] = labels
    layout["polar"]["angularaxis"]["tickmode"] = "array"
    fig.update_layout(**layout)
    return fig


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ──────────────────────────────────────────────
# RENDER: HEADER
# ──────────────────────────────────────────────

def render_app_header(df: pd.DataFrame, positions: list[str]) -> None:
    clusters = int(df[[POSITION_COL, CLUSTER_COL]].drop_duplicates().shape[0])
    competitions = int(df["competicion"].nunique()) if "competicion" in df.columns else 1
    comps_label = df["competicion"].unique() if "competicion" in df.columns else ["—"]
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-kicker">⚽ Player Profiling Dashboard · Fútbol Femenino</div>
            <div class="hero-title">Scout<span> Profiles</span> Lab</div>
            <p class="hero-copy">
                Herramienta interactiva para interpretar perfiles tácticos de jugadoras,
                comparar cada caso con su cluster y construir rankings personalizados
                con métricas ponderadas.
            </p>
            <div class="stat-strip">
                <div class="stat-chip">
                    <div class="label">Jugadoras</div>
                    <div class="value">{len(df)}</div>
                </div>
                <div class="stat-chip">
                    <div class="label">Posiciones</div>
                    <div class="value">{len(positions)}</div>
                </div>
                <div class="stat-chip">
                    <div class="label">Perfiles detectados</div>
                    <div class="value">{clusters}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Datos: `{DATA_PATH}` · {competitions} competición(es): {', '.join(str(c) for c in comps_label)}")


# ──────────────────────────────────────────────
# RENDER: SIDEBAR
# ──────────────────────────────────────────────

def render_sidebar_intro() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <strong>Scout Profiles Lab</strong>
            <span class="sub">Análisis táctico · <span class="accent">Fútbol Femenino</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# RENDER: SIMILAR PLAYER CARDS
# ──────────────────────────────────────────────

def render_similar_players_cards(similar: pd.DataFrame, fetch_photos: bool) -> None:
    cards_html = '<div class="sim-grid">'
    for _, row in similar.iterrows():
        name = str(row[PLAYER_COL])
        team = str(row[TEAM_COL])
        cluster = int(row[CLUSTER_COL])
        dist = float(row["distancia"])
        color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        avatar = player_avatar_html(name, team, size=64, border_color=color, fetch_photo=fetch_photos)

        cards_html += f"""
        <div class="sim-card">
            {avatar}
            <div class="sim-name">{name}</div>
            <div class="sim-team">{team}</div>
            <span class="sim-cluster" style="background:{color}">C{cluster}</span>
            <div class="sim-dist">dist: {dist:.2f}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# RENDER: PLAYER MODE
# ──────────────────────────────────────────────

def render_player_mode(df: pd.DataFrame, position: str) -> None:
    df_pos = df[df[POSITION_COL] == position].copy()
    metrics = metric_columns(df_pos, position)

    # Sidebar controls
    search = st.sidebar.text_input("🔍 Buscar jugadora o equipo", "")
    if search:
        text = search.lower()
        df_options = df_pos[
            df_pos[PLAYER_COL].str.lower().str.contains(text, na=False)
            | df_pos[TEAM_COL].str.lower().str.contains(text, na=False)
        ]
    else:
        df_options = df_pos

    if df_options.empty:
        st.warning("No hay jugadoras que coincidan con esa búsqueda.")
        return

    labels = {
        idx: f"{row[PLAYER_COL]} · {row[TEAM_COL]}"
        for idx, row in df_options.sort_values(PLAYER_COL).iterrows()
    }
    selected_index = st.sidebar.selectbox("Jugadora", options=list(labels), format_func=labels.get)
    player = df_pos.loc[selected_index]
    cluster = int(player[CLUSTER_COL])
    top_metrics = top_discriminant_metrics(df_pos, metrics, 20)

    fetch_photos = st.sidebar.toggle("📷 Buscar fotos de jugadoras", value=True,
                                     help="Descarga automáticamente fotos desde FotMob. "
                                          "Puede ser lento la primera vez.")
    st.sidebar.markdown("---")
    st.sidebar.metric("Jugadoras en posición", len(df_pos))
    st.sidebar.metric("En este cluster", int((df_pos[CLUSTER_COL] == cluster).sum()))

    # ── Player header card ──
    cluster_color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
    pos_label = POS_LABELS.get(position, position)
    comp = str(player.get("competicion", "—")) if "competicion" in player.index else "—"
    # --- Dentro de render_player_mode ---
    team_name = str(player[TEAM_COL])
    player_display_name = str(player[PLAYER_COL])
    
    # FIX 1: Usamos 'fetch_photos' (con S) que es como la definiste en el toggle superior
    # FIX 2: Añadimos 'team_name=team_name' para que la función sepa de qué país es la jugadora
    avatar_html = player_avatar_html(
        player_display_name, 
        team_name=team_name, # <--- Añade esto
        size=80, 
        border_color=cluster_color, 
        fetch_photo=fetch_photos # <--- Cambia fetch_photo por fetch_photos
    )
    
    # Buscar si el equipo tiene bandera en el JSON para el texto del header
    flag_html = ""
    if team_name in CUSTOM_PHOTOS:
        flag_url = CUSTOM_PHOTOS[team_name]
        flag_html = f'<img src="{flag_url}" style="height:16px; margin-left:8px; vertical-align:middle; border-radius:2px;">'
    
    # Renderizar el encabezado
    st.markdown(
        f"""
        <div class="player-header">
            {avatar_html}
            <div class="player-info">
                <div class="player-name">{player_display_name}</div>
                <div class="player-meta">
                    <span class="pos-badge">{pos_label}</span>
                    {team_name}{flag_html} · {comp}
                </div>
                <span class="cluster-badge" style="background:{cluster_color}">
                    Cluster {cluster}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    medoid = cluster_medoid_player(df_pos, cluster)
    cluster_mean = cluster_mean_row(df_pos, cluster, top_metrics)
    player_radar_vals = scaled_p05_p95_values(df_pos, player, top_metrics)
    cluster_mean_radar_vals = scaled_p05_p95_values(df_pos, cluster_mean, top_metrics)

    # ── TABS ──
    tab_radar, tab_barras, tab_cluster = st.tabs(
        ["📡 Radar individual", "📊 Percentiles detallados", "🗺️ Todos los clusters"]
    )

    with tab_radar:
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown('<div class="section-title"><span>◈</span>Radar interactivo</div>', unsafe_allow_html=True)
            st.plotly_chart(
                player_cluster_radar_figure(
                    str(player[PLAYER_COL]), cluster,
                    player_radar_vals, str(medoid[PLAYER_COL]),
                    cluster_mean_radar_vals, position,
                ),
                use_container_width=True,
            )

        with right:
            st.markdown('<div class="section-title"><span>◈</span>Lectura del cluster</div>', unsafe_allow_html=True)

            # Representativa (Añadimos el parámetro team_name)
            rep_avatar = player_avatar_html(
                str(medoid[PLAYER_COL]), 
                team_name=str(medoid[TEAM_COL]),  # <--- ESTA ES LA CORRECCIÓN
                size=52, 
                border_color="#4a6580",
                fetch_photo=fetch_photos,
            )
            st.markdown(
                f"""
                <div class="card-sm" style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                    {rep_avatar}
                    <div>
                        <div style="font-size:.7rem;color:#4a6580;text-transform:uppercase;font-weight:700;">Representativa PAM</div>
                        <div style="font-weight:700;color:#c8d8e8;font-size:.95rem;">{medoid[PLAYER_COL]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            interpretation = automatic_cluster_interpretation(df_pos, cluster, top_metrics)
            st.markdown(
                f'<div class="interp-box">{interpretation["interpretacion"]}</div>',
                unsafe_allow_html=True,
            )

            z_summary = cluster_z_summary(df_pos, cluster, top_metrics)
            strengths = z_summary.head(5)
            weaknesses = z_summary.tail(5).sort_values("z")

            st.markdown("**Fortalezas relativas del cluster**")
            pills_str = "".join(
                f'<span class="metric-pill pill-strength">↑ {clean_metric_name(r["metric"])} '
                f'<span class="z-value">z={r["z"]:+.2f}</span></span>'
                for _, r in strengths.iterrows()
            )
            st.markdown(pills_str, unsafe_allow_html=True)

            st.markdown("<br>**Debilidades relativas del cluster**", unsafe_allow_html=True)
            pills_wk = "".join(
                f'<span class="metric-pill pill-weakness">↓ {clean_metric_name(r["metric"])} '
                f'<span class="z-value">z={r["z"]:+.2f}</span></span>'
                for _, r in weaknesses.iterrows()
            )
            st.markdown(pills_wk, unsafe_allow_html=True)

    with tab_barras:
        st.markdown('<div class="section-title"><span>◈</span>Percentiles vs media del cluster</div>', unsafe_allow_html=True)
        player_pct = {m: percentile(df_pos[m], player[m]) for m in top_metrics}
        cluster_pct = cluster_percentiles(df_pos, cluster, top_metrics)
        st.plotly_chart(
            player_cluster_bar_figure(str(player[PLAYER_COL]), cluster, player_pct, cluster_pct),
            use_container_width=True,
        )

    with tab_cluster:
        st.markdown('<div class="section-title"><span>◈</span>Visión global de todos los clusters</div>', unsafe_allow_html=True)
        st.plotly_chart(cluster_radar_figure(df_pos, metrics, position), use_container_width=True)
        with st.expander("Tabla: interpretación de todos los clusters"):
            st.dataframe(
                all_cluster_interpretations(df_pos, top_metrics).rename(
                    columns={"cluster": "Cluster", "familia": "Familia", "interpretacion": "Interpretación",
                             "fortalezas": "Fortalezas", "debilidades": "Debilidades", "n": "N"}
                ),
                use_container_width=True, hide_index=True,
            )

    # ── Jugadoras parecidas ──
    st.markdown("---")
    st.markdown('<div class="section-title"><span>◈</span>Jugadoras más similares</div>', unsafe_allow_html=True)
    similar = similar_players(df_pos, player, top_metrics, 8)
    if not similar.empty:
        render_similar_players_cards(similar, fetch_photos=fetch_photos)
    else:
        st.info("No hay suficientes jugadoras para calcular similares.")


# ──────────────────────────────────────────────
# RENDER: PROFILE MODE
# ──────────────────────────────────────────────

def render_profile_mode(df: pd.DataFrame, position: str) -> None:
    df_pos = df[df[POSITION_COL] == position].copy()
    metric_scope = st.radio(
        "Conjunto de métricas",
        ["Métricas del clustering", "Todas las métricas derivadas"],
        horizontal=True,
    )
    if metric_scope == "Métricas del clustering":
        metrics = metric_columns(df_pos, position)
    else:
        metrics = all_derived_metric_columns(df_pos)

    suggested = top_discriminant_metrics(df_pos, metrics, 18)

    st.markdown('<div class="section-title"><span>◈</span>Constructor de perfil</div>', unsafe_allow_html=True)
    st.caption(
        "Peso positivo → busca valores altos. Peso negativo → busca valores bajos. Cero → no se usa."
    )

    selected_metrics = st.multiselect(
        "Métricas candidatas",
        options=metrics, default=suggested[:10], format_func=clean_metric_name,
    )

    weights: dict[str, int] = {}
    if selected_metrics:
        cols = st.columns(3)
        for i, metric in enumerate(selected_metrics):
            with cols[i % 3]:
                weights[metric] = st.slider(
                    clean_metric_name(metric), min_value=-5, max_value=5, value=1, step=1,
                    key=f"w_{position}_{metric}",
                )

    scores = profile_score(df_pos, weights)
    if scores.empty:
        st.info("Selecciona al menos una métrica con peso distinto de cero.")
        return

    fetch_photos = st.sidebar.toggle("📷 Buscar fotos de jugadoras", value=True)
    st.sidebar.markdown("---")
    st.sidebar.metric("Jugadoras en posición", len(df_pos))

    st.markdown("---")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-title"><span>◈</span>Radar del perfil buscado</div>', unsafe_allow_html=True)
        st.plotly_chart(profile_radar_figure(df_pos, weights, scores), use_container_width=True)

    with right:
        st.markdown('<div class="section-title"><span>◈</span>Top 15 jugadoras</div>', unsafe_allow_html=True)
        top_scores = scores.head(15).sort_values("score")
        fig = px.bar(
            top_scores, x="score", y=PLAYER_COL, orientation="h",
            color="score", color_continuous_scale=["#1a2e45", "#2A9D8F", "#E63946"],
            labels={"score": "Score", PLAYER_COL: "Jugadora"},
            hover_data=[TEAM_COL, CLUSTER_COL],
        )
        fig.update_layout(
            height=580, margin=dict(l=20, r=20, t=20, b=30),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1a2e45", tickfont=dict(color="#7a9bb5")),
            yaxis=dict(tickfont=dict(color="#c8d8e8")),
            **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title"><span>◈</span>Tabla completa de scores</div>', unsafe_allow_html=True)
    st.dataframe(
        scores.head(30).rename(
            columns={PLAYER_COL: "Jugadora", TEAM_COL: "Equipo",
                     POSITION_COL: "Posición", CLUSTER_COL: "Cluster", "score": "Score"}
        ),
        use_container_width=True, hide_index=True,
    )

    # Top match card
    if not scores.empty:
        best = scores.iloc[0]
        best_cluster = int(best[CLUSTER_COL])
        best_color = CLUSTER_COLORS[best_cluster % len(CLUSTER_COLORS)]
        best_avatar = player_avatar_html(str(best[PLAYER_COL]), size=70, border_color=best_color,
                                         fetch_photo=fetch_photos)
        st.markdown(
            f"""
            <div class="card" style="margin-top:16px;display:flex;align-items:center;gap:16px;">
                {best_avatar}
                <div>
                    <div style="font-size:.7rem;color:#4a6580;text-transform:uppercase;font-weight:700;">
                        🏆 Mejor encaje con el perfil buscado
                    </div>
                    <div style="font-size:1.4rem;font-weight:800;color:#e2e8f2;">{best[PLAYER_COL]}</div>
                    <div style="color:#6a8aa5;font-size:.88rem;">{best[TEAM_COL]} ·
                        <span class="cluster-badge" style="background:{best_color};font-size:.75rem;">
                            C{best_cluster}
                        </span> ·
                        <span style="font-family:'Space Mono',monospace;color:#2A9D8F;">
                            {best['score']:.1f} pts
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# RENDER: GLOSSARY & METHODOLOGY
# ──────────────────────────────────────────────

def render_glossary() -> None:
    with st.expander("📖 Glosario y funcionamiento"):
        st.markdown(
            """
            <div class="glossary-line"><span class="glossary-term">Modo jugadora.</span>
            Selecciona posición y jugadora. La app muestra su cluster, un radar frente al perfil típico,
            una interpretación automática y jugadoras similares con fotos.</div>

            <div class="glossary-line"><span class="glossary-term">Modo perfil creado.</span>
            Elige métricas y pesos. Pesos positivos buscan valores altos; negativos, valores bajos.
            El score indica el encaje de cada jugadora con ese perfil.</div>

            <div class="glossary-line"><span class="glossary-term">Cluster.</span>
            Grupo de jugadoras con comportamiento estadístico parecido dentro de la misma posición.</div>

            <div class="glossary-line"><span class="glossary-term">Perfil típico (radar gris).</span>
            Media del cluster, normalizada p05–p95, para representar el patrón medio del grupo.
            El nombre entre paréntesis es la representativa PAM exportada.</div>

            <div class="glossary-line"><span class="glossary-term">Radar p05–p95.</span>
            Cada métrica se escala entre el percentil 5 y el 95 de la posición,
            reduciendo el efecto de valores extremos y permitiendo comparar métricas con unidades distintas.</div>

            <div class="glossary-line"><span class="glossary-term">Fotos de jugadoras.</span>
            Se descargan automáticamente desde FotMob. Se almacenan en caché local (<code>.photo_cache/</code>)
            para evitar peticiones repetidas. Si una jugadora no aparece en la búsqueda, se muestra un avatar
            con sus iniciales.</div>
            """,
            unsafe_allow_html=True,
        )


def render_methodology_note() -> None:
    with st.expander("🔬 Ideas para reforzar la metodología"):
        st.markdown(
            """
            Una silueta baja no invalida automáticamente el análisis. En rendimiento deportivo los perfiles suelen
            solaparse porque las jugadoras no pertenecen a tipos puros, sino a un continuo táctico. Para defenderlo:

            - **Interpretabilidad**: medias por cluster, z-scores, medoides y ejemplos conocidos.
            - **Estabilidad**: repite el clustering con bootstrap o distintas temporadas.
            - **PCA / UMAP**: solo como apoyo explicativo, no como prueba principal.
            - **Validación cualitativa**: si una jugadora conocida cae en el perfil correcto, eso suma.
            - **Modo supervisado**: documenta que buscas encaje con un perfil definido, no clusters naturales.
            """,
        )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    inject_css()
    df = load_data()
    if df is None:
        return

    positions = sorted(df[POSITION_COL].dropna().unique().tolist())
    render_sidebar_intro()

    st.sidebar.markdown("### ⚙️ Configuración")
    position = st.sidebar.selectbox(
        "Posición",
        positions,
        format_func=lambda p: f"{POS_LABELS.get(p, p)} ({p})",
    )
    mode = st.sidebar.radio(
        "Modo de análisis",
        ["A partir de una jugadora", "A partir de un perfil creado"],
        help="Elige si quieres analizar una jugadora concreta o construir un perfil propio.",
    )

    render_app_header(df, positions)

    if mode == "A partir de una jugadora":
        render_player_mode(df, position)
    else:
        render_profile_mode(df, position)

    st.markdown("---")
    render_glossary()
    render_methodology_note()

    st.markdown(
        "<p style='text-align:center;color:#1a2e45;font-size:.75rem;margin-top:24px;'>"
        "Scout Profiles Lab · Fútbol Femenino · TFG Analysis Tool</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
