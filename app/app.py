from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DATA_PATH = Path("data/perfiles_finales.csv")
POSITION_COL = "grupo_pos_clustering"
CLUSTER_COL = "cluster_final"
PLAYER_COL = "player_name"
TEAM_COL = "team_name"
ID_COL = "player_id"

METADATA_COLS = {
    ID_COL,
    PLAYER_COL,
    TEAM_COL,
    "competicion",
    "posicion_principal",
    "grupo_pos",
    POSITION_COL,
    CLUSTER_COL,
    "cluster",
    "cluster_fa",
    "cluster_pca",
    "cluster_hier",
    "es_medoide",
    "nombre_medoide_cluster",
}

POS_METRICS = {
    "CB": [
        "pct_pases",
        "pct_pases_largos",
        "pct_pases_prog",
        "pct_pases_bajo_presion",
        "pases_ult_tercio_p90",
        "carries_prog_p90",
        "ratio_intercepciones_vs_entradas",
        "duelos_ter_ganados_p90",
        "despejes_p90",
        "pct_aereos",
        "pct_duelos_total",
        "recuperaciones_p90",
        "acciones_defensivas_campo_rival_p90",
    ],
    "LAT": [
        "pct_toques_en_campo_rival",
        "centros_p90",
        "pases_al_area_p90",
        "ratio_centros_vs_pases_al_area",
        "deep_progressions_p90",
        "xa_real_p90",
        "carries_prog_p90",
        "pct_regates",
        "ratio_intercepciones_vs_entradas",
        "acciones_defensivas_p90",
        "duelos_ter_ganados_p90",
        "despejes_p90",
        "pct_duelos_total",
        "recuperaciones_p90",
    ],
    "MCD": [
        "distancia_media_pases",
        "pct_pases_prog",
        "pct_pases_bajo_presion",
        "pases_progresivos_p90",
        "pases_largos_p90",
        "ratio_intercepciones_vs_entradas",
        "intercepciones_p90",
        "recuperaciones_p90",
        "acciones_agresivas_p90",
        "presiones_p90",
    ],
    "MC": [
        "carries_prog_p90",
        "velocidad_cond_m_s",
        "pct_pases",
        "pases_completados_p90",
        "pct_pases_prog",
        "distancia_media_pases",
        "pases_bajo_presion_p90",
        "xa_real_p90",
        "presiones_p90",
        "intercepciones_p90",
        "recuperaciones_p90",
        "pct_duelos_total",
        "pases_al_area_p90",
        "through_balls_p90",
        "pct_toques_en_area",
        "tiros_p90",
    ],
    "EXT": [
        "regates_p90",
        "pct_regates",
        "carries_prog_p90",
        "ratio_centros_vs_pases_al_area",
        "centros_p90",
        "pct_pases",
        "pases_completados_p90",
        "pct_toques_en_area",
        "tiros_p90",
        "xg_por_tiro",
        "xa_real_p90",
        "pases_clave_p90",
    ],
    "DEL": [
        "ratio_tiros_vs_pases",
        "xg_por_tiro",
        "distancia_media_tiros",
        "pct_toques_en_area",
        "tiros_puerta_p90",
        "pases_completados_p90",
        "pct_pases",
        "pct_pases_prog",
        "pases_clave_p90",
        "xa_real_p90",
        "through_balls_p90",
        "recibidos_de_espaldas_p90",
        "presiones_p90",
        "aereos_ganados_p90",
    ],
}

DERIVED_METRIC_PREFIXES = (
    "pct_",
    "ratio_",
    "rel_",
    "obv_",
)

DERIVED_METRIC_SUFFIXES = (
    "_p90",
)

DERIVED_METRIC_NAMES = {
    "xg_por_tiro",
    "distancia_media_pases",
    "distancia_media_tiros",
    "posicion_media_x",
    "posicion_media_y",
    "std_posicion_x",
    "std_posicion_y",
    "velocidad_cond_m_s",
    "challenge_ratio",
}


st.set_page_config(
    page_title="Scout Profiles Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #f8fafc 48%, #ffffff 100%);
        }
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2.4rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8dee8;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.25rem;
            line-height: 1.08;
            color: #17202a;
        }
        h2, h3 {
            color: #17202a;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 12px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(22, 34, 51, 0.04);
        }
        div[data-testid="stMetricValue"] {
            color: #17202a;
            font-weight: 800;
        }
        .profile-pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 8px;
            color: white;
            background: #2357c6;
            font-weight: 800;
        }
        .hero {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 22px 24px;
            background:
                linear-gradient(135deg, rgba(35,87,198,.10), rgba(15,143,140,.08)),
                #ffffff;
            box-shadow: 0 14px 36px rgba(22, 34, 51, 0.06);
            margin-bottom: 18px;
        }
        .hero-kicker {
            color: #0f8f8c;
            text-transform: uppercase;
            font-weight: 850;
            font-size: .78rem;
            letter-spacing: .06em;
            margin-bottom: 8px;
        }
        .hero-title {
            font-size: 2.25rem;
            line-height: 1.05;
            margin: 0 0 8px 0;
            font-weight: 850;
            color: #17202a;
        }
        .hero-copy {
            max-width: 920px;
            color: #526070;
            font-size: 1rem;
            margin: 0;
        }
        .section-card {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(22, 34, 51, 0.045);
        }
        .soft-box {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 92px;
        }
        .small-muted {
            color: #667085;
            font-size: 0.88rem;
        }
        .sidebar-brand {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 14px;
            background: #f8fafc;
            margin-bottom: 14px;
        }
        .sidebar-brand strong {
            display: block;
            font-size: 1.05rem;
            margin-bottom: 4px;
            color: #17202a;
        }
        .sidebar-brand span {
            color: #667085;
            font-size: .86rem;
        }
        .info-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 6px;
        }
        .info-chip {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 10px 12px;
            background: #ffffff;
        }
        .info-chip span {
            display: block;
            color: #667085;
            font-size: .76rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .info-chip strong {
            display: block;
            color: #17202a;
            font-size: 1rem;
            margin-top: 3px;
        }
        .glossary-term {
            font-weight: 800;
            color: #17202a;
        }
        .glossary-copy {
            color: #526070;
            margin-bottom: .7rem;
        }
        @media (max-width: 900px) {
            .info-strip {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(file_mtime: float | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        st.error(
            "No encuentro `data/perfiles_finales.csv`. Coloca el CSV exportado desde Colab "
            "en esa ruta dentro del repositorio."
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


def metric_columns(df: pd.DataFrame, position: str | None = None) -> list[str]:
    source = df[df[POSITION_COL] == position] if position else df
    numeric_cols = source.select_dtypes(include=np.number).columns.tolist()

    if position in POS_METRICS:
        return [metric for metric in POS_METRICS[position] if metric in numeric_cols]

    metrics = [
        col
        for col in numeric_cols
        if col not in METADATA_COLS and not col.startswith("cluster")
        and (
            col.endswith(DERIVED_METRIC_SUFFIXES)
            or col.startswith(DERIVED_METRIC_PREFIXES)
            or col in DERIVED_METRIC_NAMES
        )
    ]
    return sorted(metrics)


def clean_metric_name(metric: str) -> str:
    return (
        metric.replace("_p90", "")
        .replace("pct_", "% ")
        .replace("_", " ")
    )


def metric_family(metric: str) -> str:
    metric_lower = metric.lower()
    if any(token in metric_lower for token in ["xg", "tiros", "goles", "area", "espaldas"]):
        return "finalización y presencia en área"
    if any(token in metric_lower for token in ["pases", "through", "xa", "asistencias", "centros"]):
        return "pase y creación"
    if any(token in metric_lower for token in ["carries", "regates", "progressions", "condu", "velocidad"]):
        return "conducción y progresión"
    if any(token in metric_lower for token in ["presiones", "recuperaciones", "intercepciones", "defensivas", "despejes", "duelos", "aereos"]):
        return "defensa y recuperación"
    if any(token in metric_lower for token in ["posicion", "std_"]):
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
    return [metric for metric, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:limit]]


def player_percentiles(df_pos: pd.DataFrame, player: pd.Series, metrics: Iterable[str]) -> dict[str, float]:
    return {metric: percentile(df_pos[metric], player[metric]) for metric in metrics}


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
    result = {}
    for metric in metrics:
        mean_value = df_cluster[metric].mean()
        result[metric] = percentile(df_pos[metric], mean_value)
    return result


def cluster_medoid_player(df_pos: pd.DataFrame, cluster: int) -> pd.Series:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    if df_cluster.empty:
        return df_pos.iloc[0]

    if "es_medoide" in df_cluster.columns:
        medoid_mask = df_cluster["es_medoide"].astype(str).str.lower().isin(["true", "1", "si", "sí"])
        if medoid_mask.any():
            return df_cluster[medoid_mask].iloc[0]

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


def all_derived_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return sorted(
        col
        for col in numeric_cols
        if col not in METADATA_COLS
        and not col.startswith("cluster")
        and (
            col.endswith(DERIVED_METRIC_SUFFIXES)
            or col.startswith(DERIVED_METRIC_PREFIXES)
            or col in DERIVED_METRIC_NAMES
        )
    )


def radar_angles(n_metrics: int) -> list[float]:
    return [i * 360.0 / n_metrics for i in range(n_metrics)]


def player_cluster_radar_figure(
    player_name: str,
    cluster: int,
    player_values_by_metric: dict[str, float],
    representative_name: str,
    representative_values_by_metric: dict[str, float],
    position: str,
) -> go.Figure:
    metrics = list(player_values_by_metric.keys())
    labels = [clean_metric_name(metric) for metric in metrics]
    angles = radar_angles(len(metrics))
    closed_angles = angles + [360.0]
    player_values = [player_values_by_metric[metric] for metric in metrics]
    representative_values = [representative_values_by_metric[metric] for metric in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=player_values + player_values[:1],
            theta=closed_angles,
            mode="lines+markers",
            fill="toself",
            name=player_name,
            line=dict(color="#E74C3C", width=3),
            marker=dict(size=6),
            text=labels + labels[:1],
            hovertemplate="%{text}<br>Escala p05-p95: %{r:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=representative_values + representative_values[:1],
            theta=closed_angles,
            mode="lines+markers",
            fill="toself",
            name=f"Perfil Típico Cluster {cluster} ({representative_name[:24]})",
            line=dict(color="#95A5A6", width=2.5),
            marker=dict(size=5),
            text=labels + labels[:1],
            hovertemplate="%{text}<br>Escala p05-p95: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{player_name} — {position} (cluster {cluster})",
        height=650,
        margin=dict(l=70, r=70, t=70, b=45),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.15, 0.5, 0.85],
                ticktext=["p05", "p50", "p95"],
                gridcolor="#d8dee8",
                tickfont=dict(size=10, color="#667085"),
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=angles,
                ticktext=labels,
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=10, color="#667085"),
                gridcolor="#d8dee8",
            ),
        ),
        legend=dict(orientation="h", y=1.08, x=0.62, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def cluster_radar_figure(df_pos: pd.DataFrame, metrics: list[str], position: str, top_n: int = 15) -> go.Figure:
    medias_clusters = df_pos.groupby(CLUSTER_COL)[metrics].mean(numeric_only=True).sort_index()
    selected_metrics = top_discriminant_metrics(df_pos, metrics, top_n)
    if not selected_metrics:
        selected_metrics = metrics[:top_n]

    norm = pd.DataFrame(index=medias_clusters.index, columns=selected_metrics, dtype=float)
    for metric in selected_metrics:
        norm[metric] = [
            percentile(df_pos[metric], medias_clusters.loc[cluster, metric])
            for cluster in medias_clusters.index
        ]

    representative_names = representative_players(df_pos, selected_metrics)
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#8E44AD", "#2357C6"]
    labels = [clean_metric_name(metric) for metric in selected_metrics]

    n_clusters = len(norm)
    cols = min(3, n_clusters)
    rows = int(np.ceil(n_clusters / cols))
    specs = [[{"type": "polar"} for _ in range(cols)] for _ in range(rows)]
    titles = [
        f"Cluster {cluster}<br><sup>{representative_names.get(cluster, 'Representativa no disponible')}</sup>"
        for cluster in norm.index
    ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    for i, cluster in enumerate(norm.index):
        row = i // cols + 1
        col = i % cols + 1
        values = norm.loc[cluster].astype(float).tolist()
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=labels + labels[:1],
                fill="toself",
                name=f"Cluster {cluster}",
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.78,
                hovertemplate="%{theta}<br>Percentil %{r:.0f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )

        polar_name = "polar" if i == 0 else f"polar{i + 1}"
        fig.layout[polar_name].update(
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], ticktext=["", "50%", ""], showline=False),
            angularaxis=dict(tickfont=dict(size=10, color="#333333")),
        )

    fig.update_layout(
        title=f"Perfiles tácticos por percentil · Posición: {position}",
        height=380 * rows + 90,
        showlegend=False,
        margin=dict(l=30, r=30, t=90, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def representative_players(df_pos: pd.DataFrame, metrics: list[str]) -> dict[int, str]:
    if not metrics:
        return {}

    result: dict[int, str] = {}
    values = df_pos[metrics].apply(pd.to_numeric, errors="coerce").fillna(df_pos[metrics].median())
    std = values.std(ddof=0).replace(0, 1)
    z_values = (values - values.mean()) / std

    for cluster, cluster_df in df_pos.groupby(CLUSTER_COL):
        cluster_z = z_values.loc[cluster_df.index]
        centroid = cluster_z.mean()
        distances = np.sqrt(((cluster_z - centroid) ** 2).sum(axis=1))
        representative_index = distances.idxmin()
        result[int(cluster)] = str(df_pos.loc[representative_index, PLAYER_COL])

    return result


def player_cluster_bar_figure(
    player_name: str,
    cluster: int,
    player_pct: dict[str, float],
    cluster_pct: dict[str, float],
) -> go.Figure:
    metrics = list(player_pct.keys())
    labels = [clean_metric_name(metric) for metric in metrics]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[player_pct[metric] for metric in metrics],
            name=player_name,
            orientation="h",
            marker_color="#2357c6",
            hovertemplate="%{y}<br>Percentil %{x:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[cluster_pct[metric] for metric in metrics],
            name=f"Media cluster {cluster}",
            orientation="h",
            marker_color="#0f8f8c",
            hovertemplate="%{y}<br>Percentil %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        height=max(420, 34 * len(metrics) + 120),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h"),
        margin=dict(l=130, r=25, t=25, b=45),
    )
    return fig


def cluster_z_summary(df_pos: pd.DataFrame, cluster: int, metrics: Iterable[str]) -> pd.DataFrame:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    rows = []
    for metric in metrics:
        global_mean = df_pos[metric].mean()
        global_std = df_pos[metric].std(ddof=0)
        cluster_mean = df_cluster[metric].mean()
        z_value = 0 if global_std == 0 or pd.isna(global_std) else (cluster_mean - global_mean) / global_std
        rows.append({"metric": metric, "z": z_value})
    return pd.DataFrame(rows).sort_values("z", ascending=False)


def automatic_cluster_interpretation(df_pos: pd.DataFrame, cluster: int, metrics: list[str]) -> dict[str, object]:
    summary = cluster_z_summary(df_pos, cluster, metrics)
    highs = summary.head(4)
    lows = summary.tail(3).sort_values("z")
    cluster_size = int((df_pos[CLUSTER_COL] == cluster).sum())

    families = highs["metric"].map(metric_family).value_counts()
    dominant_family = families.index[0] if not families.empty else "perfil mixto"
    high_names = [clean_metric_name(metric) for metric in highs["metric"].tolist()]
    low_names = [clean_metric_name(metric) for metric in lows["metric"].tolist()]

    text = (
        f"Cluster {cluster}: perfil con tendencia principal hacia {dominant_family}. "
        f"Destaca por {', '.join(high_names[:3])}. "
        f"Sus valores relativamente más bajos aparecen en {', '.join(low_names[:2])}. "
        f"Contiene {cluster_size} jugadoras."
    )

    return {
        "cluster": cluster,
        "familia": dominant_family,
        "interpretacion": text,
        "fortalezas": ", ".join(high_names),
        "debilidades": ", ".join(low_names),
        "n": cluster_size,
    }


def all_cluster_interpretations(df_pos: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = [
        automatic_cluster_interpretation(df_pos, int(cluster), metrics)
        for cluster in sorted(df_pos[CLUSTER_COL].dropna().unique())
    ]
    return pd.DataFrame(rows)


def similar_players(df_pos: pd.DataFrame, player: pd.Series, metrics: list[str], limit: int = 10) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    values = df_pos[metrics].apply(pd.to_numeric, errors="coerce").fillna(df_pos[metrics].median())
    std = values.std(ddof=0).replace(0, 1)
    z_values = (values - values.mean()) / std
    player_index = player.name
    distances = np.sqrt(((z_values - z_values.loc[player_index]) ** 2).sum(axis=1))
    result = df_pos.copy()
    result["distancia"] = distances
    result = result[result.index != player_index]
    return result.sort_values("distancia").head(limit)


def profile_score(df_pos: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    active = {metric: weight for metric, weight in weights.items() if weight != 0}
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


def profile_radar_figure(
    df_pos: pd.DataFrame,
    weights: dict[str, int],
    scores: pd.DataFrame,
) -> go.Figure:
    active = {metric: weight for metric, weight in weights.items() if weight != 0}
    metrics = list(active.keys())
    labels = [clean_metric_name(metric) for metric in metrics]
    angles = radar_angles(len(metrics))
    closed_angles = angles + [360.0]

    desired_values = [
        0.5 + (np.sign(active[metric]) * abs(active[metric]) / 5.0 * 0.35)
        for metric in metrics
    ]

    best_values = []
    best_name = "Mejor encaje"
    if not scores.empty:
        best_row = scores.iloc[0]
        best_name = str(best_row[PLAYER_COL])
        best_match = df_pos[
            (df_pos[PLAYER_COL] == best_row[PLAYER_COL])
            & (df_pos[TEAM_COL] == best_row[TEAM_COL])
        ]
        if best_match.empty:
            best_match = df_pos[df_pos[PLAYER_COL] == best_row[PLAYER_COL]]
        if not best_match.empty:
            best_values_by_metric = scaled_p05_p95_values(df_pos, best_match.iloc[0], metrics)
            best_values = [best_values_by_metric[metric] for metric in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=desired_values + desired_values[:1],
            theta=closed_angles,
            mode="lines+markers",
            fill="toself",
            name="Perfil buscado",
            line=dict(color="#2357c6", width=3),
            marker=dict(size=6),
            text=labels + labels[:1],
            hovertemplate="%{text}<br>Objetivo: %{r:.2f}<extra></extra>",
        )
    )
    if best_values:
        fig.add_trace(
            go.Scatterpolar(
                r=best_values + best_values[:1],
                theta=closed_angles,
                mode="lines+markers",
                fill="toself",
                name=f"Mejor encaje: {best_name[:24]}",
                line=dict(color="#0f8f8c", width=2.5),
                marker=dict(size=5),
                text=labels + labels[:1],
                hovertemplate="%{text}<br>Escala p05-p95: %{r:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=50, r=50, t=35, b=45),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.15, 0.5, 0.85],
                ticktext=["bajo", "medio", "alto"],
                gridcolor="#d8dee8",
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=angles,
                ticktext=labels,
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=10, color="#667085"),
                gridcolor="#d8dee8",
            ),
        ),
        legend=dict(orientation="h", y=-0.08),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_player_mode(df: pd.DataFrame, position: str) -> None:
    df_pos = df[df[POSITION_COL] == position].copy()
    metrics = metric_columns(df_pos, position)

    search = st.sidebar.text_input("Buscar jugadora o equipo", "")
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
        idx: f"{row[PLAYER_COL]} · {row[TEAM_COL]} · C{row[CLUSTER_COL]}"
        for idx, row in df_options.sort_values(PLAYER_COL).iterrows()
    }
    selected_index = st.sidebar.selectbox("Jugadora", options=list(labels), format_func=labels.get)
    player = df_pos.loc[selected_index]
    cluster = int(player[CLUSTER_COL])
    top_metrics = top_discriminant_metrics(df_pos, metrics, 20)

    st.markdown(
        f"### {player[PLAYER_COL]}  \n"
        f"<span class='small-muted'>{player[TEAM_COL]} · {position}</span> "
        f"<span class='profile-pill'>Cluster {cluster}</span>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Jugadoras en posición", len(df_pos))
    col_b.metric("Jugadoras en cluster", int((df_pos[CLUSTER_COL] == cluster).sum()))
    col_c.metric("Métricas del radar", len(top_metrics))

    medoid = cluster_medoid_player(df_pos, cluster)
    cluster_mean = cluster_mean_row(df_pos, cluster, top_metrics)
    player_radar_values = scaled_p05_p95_values(df_pos, player, top_metrics)
    cluster_mean_radar_values = scaled_p05_p95_values(df_pos, cluster_mean, top_metrics)

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Radar interactivo")
        st.plotly_chart(
            player_cluster_radar_figure(
                str(player[PLAYER_COL]),
                cluster,
                player_radar_values,
                str(medoid[PLAYER_COL]),
                cluster_mean_radar_values,
                position,
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("Lectura del cluster")
        st.write(f"**Representativa PAM:** {medoid[PLAYER_COL]}")
        st.write("**Radar gris:** media del cluster, como perfil típico.")
        interpretation = automatic_cluster_interpretation(df_pos, cluster, top_metrics)
        st.markdown("**Interpretación automática**")
        st.write(interpretation["interpretacion"])

        z_summary = cluster_z_summary(df_pos, cluster, top_metrics)
        strengths = z_summary.head(4)
        weaknesses = z_summary.tail(4).sort_values("z")

        st.markdown("**Fortalezas relativas**")
        for _, row in strengths.iterrows():
            st.write(f"- {clean_metric_name(row['metric'])}: z {row['z']:+.2f}")

        st.markdown("**Debilidades relativas**")
        for _, row in weaknesses.iterrows():
            st.write(f"- {clean_metric_name(row['metric'])}: z {row['z']:+.2f}")

        st.markdown("**Jugadoras parecidas**")
        similar = similar_players(df_pos, player, top_metrics, 8)
        st.dataframe(
            similar[[PLAYER_COL, TEAM_COL, CLUSTER_COL, "distancia"]].rename(
                columns={
                    PLAYER_COL: "Jugadora",
                    TEAM_COL: "Equipo",
                    CLUSTER_COL: "Cluster",
                    "distancia": "Distancia",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Interpretación automática de todos los clusters de la posición"):
        st.dataframe(
            all_cluster_interpretations(df_pos, metrics).rename(
                columns={
                    "cluster": "Cluster",
                    "familia": "Familia dominante",
                    "interpretacion": "Interpretación",
                    "fortalezas": "Fortalezas",
                    "debilidades": "Debilidades",
                    "n": "N",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_profile_mode(df: pd.DataFrame, position: str) -> None:
    df_pos = df[df[POSITION_COL] == position].copy()
    metric_scope = st.radio(
        "Conjunto de métricas",
        [
            "Métricas del clustering de esta posición",
            "Todas las métricas derivadas",
        ],
        horizontal=True,
        help=(
            "La primera opción es la más coherente con la metodología del clustering. "
            "La segunda sirve para perfiles personalizados más exploratorios."
        ),
    )
    if metric_scope == "Métricas del clustering de esta posición":
        metrics = metric_columns(df_pos, position)
    else:
        metrics = all_derived_metric_columns(df_pos)
    suggested = top_discriminant_metrics(df_pos, metrics, 18)

    st.subheader("Crear un perfil con métricas y pesos")
    st.caption(
        "Recomendado: usa las métricas del clustering para mantener coherencia metodológica. "
        "El modo avanzado permite explorar otras métricas normalizadas o derivadas. "
        "Peso positivo: busca valores altos. Peso negativo: busca valores bajos. Cero: no se usa."
    )

    selected_metrics = st.multiselect(
        "Métricas candidatas",
        options=metrics,
        default=suggested[:10],
        format_func=clean_metric_name,
    )

    weights: dict[str, int] = {}
    if selected_metrics:
        cols = st.columns(2)
        for i, metric in enumerate(selected_metrics):
            with cols[i % 2]:
                weights[metric] = st.slider(
                    clean_metric_name(metric),
                    min_value=-5,
                    max_value=5,
                    value=1,
                    step=1,
                    key=f"weight_{position}_{metric}",
                )

    scores = profile_score(df_pos, weights)
    if scores.empty:
        st.info("Selecciona al menos una métrica con peso distinto de cero.")
        return

    st.subheader("Radar del perfil buscado")
    st.plotly_chart(profile_radar_figure(df_pos, weights, scores), use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        top_scores = scores.head(15).sort_values("score")
        fig = px.bar(
            top_scores,
            x="score",
            y=PLAYER_COL,
            orientation="h",
            color="score",
            color_continuous_scale=["#b86b00", "#0f8f8c", "#2357c6"],
            labels={"score": "Score", PLAYER_COL: "Jugadora"},
            hover_data=[TEAM_COL, CLUSTER_COL],
        )
        fig.update_layout(height=620, margin=dict(l=20, r=20, t=25, b=35), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.dataframe(
            scores.head(30).rename(
                columns={
                    PLAYER_COL: "Jugadora",
                    TEAM_COL: "Equipo",
                    POSITION_COL: "Posición",
                    CLUSTER_COL: "Cluster",
                    "score": "Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_methodology_note() -> None:
    with st.expander("Ideas para reforzar la metodología del TFG"):
        st.write(
            """
            Una silueta baja no invalida automáticamente el análisis. En rendimiento deportivo los perfiles suelen
            solaparse porque las jugadoras no pertenecen a tipos puros, sino a un continuo táctico. Para defenderlo mejor:

            - Complementa silueta con interpretabilidad: medias por cluster, z-scores, medoides y ejemplos conocidos.
            - Mide estabilidad: repite clustering con bootstrap o distintas temporadas y mira si los perfiles se mantienen.
            - Evalúa separación visual con PCA/UMAP solo como apoyo explicativo, no como prueba principal.
            - Usa esta app como validación cualitativa: si una jugadora conocida cae en un perfil coherente, eso suma.
            - En el modo supervisado, documenta que ya no buscas clusters naturales, sino encaje con un perfil definido.
            """
        )


def render_app_header(df: pd.DataFrame, positions: list[str]) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Player profiling dashboard</div>
            <div class="hero-title">Scout Profiles Lab</div>
            <p class="hero-copy">
                Herramienta interactiva para interpretar perfiles tácticos de jugadoras, comparar cada caso
                con su cluster y construir rankings personalizados a partir de métricas ponderadas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    clusters = int(df[[POSITION_COL, CLUSTER_COL]].drop_duplicates().shape[0])
    competitions = int(df["competicion"].nunique()) if "competicion" in df.columns else 1
    st.markdown(
        f"""
        <div class="info-strip">
            <div class="info-chip"><span>Jugadoras</span><strong>{len(df)}</strong></div>
            <div class="info-chip"><span>Posiciones</span><strong>{len(positions)}</strong></div>
            <div class="info-chip"><span>Perfiles detectados</span><strong>{clusters}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Datos cargados desde `{DATA_PATH}` · competiciones: {competitions}")


def render_sidebar_intro() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <strong>Scout Profiles Lab</strong>
            <span>Explora perfiles por clustering o crea un perfil propio con pesos.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glossary() -> None:
    with st.expander("Glosario y funcionamiento de la herramienta"):
        st.markdown(
            """
            <div class="glossary-copy">
                <span class="glossary-term">Modo jugadora.</span>
                Selecciona una posición y una jugadora. La app muestra su cluster, un radar frente al perfil típico
                del cluster, una interpretación automática y jugadoras similares.
            </div>
            <div class="glossary-copy">
                <span class="glossary-term">Modo perfil creado.</span>
                El usuario elige métricas y pesos. Pesos positivos buscan valores altos; pesos negativos buscan valores
                bajos. El score final indica el grado de encaje de cada jugadora con ese perfil.
            </div>
            <div class="glossary-copy">
                <span class="glossary-term">Cluster.</span>
                Grupo de jugadoras con comportamiento estadístico parecido dentro de una misma posición.
            </div>
            <div class="glossary-copy">
                <span class="glossary-term">Perfil típico.</span>
                En el radar gris se usa la media del cluster, normalizada como en el notebook, para representar el
                patrón medio del grupo. El nombre entre paréntesis es la representativa PAM exportada.
            </div>
            <div class="glossary-copy">
                <span class="glossary-term">Radar p05-p95.</span>
                Cada métrica se escala entre el percentil 5 y el percentil 95 de la posición. Así se reduce el efecto
                de valores extremos y se comparan métricas con unidades distintas.
            </div>
            <div class="glossary-copy">
                <span class="glossary-term">Interpretación automática.</span>
                Resume qué métricas están por encima o por debajo de la media de la posición y asigna una familia
                dominante: pase, defensa, progresión, finalización o posicionamiento.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_css()
    df = load_data()
    if df is None:
        return

    positions = sorted(df[POSITION_COL].dropna().unique().tolist())
    render_sidebar_intro()
    position = st.sidebar.selectbox("Posición", positions)
    mode = st.sidebar.radio(
        "Modo de análisis",
        ["A partir de una jugadora", "A partir de un perfil creado"],
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Jugadoras cargadas", len(df))
    st.sidebar.metric("Posiciones", len(positions))

    render_app_header(df, positions)

    if mode == "A partir de una jugadora":
        render_player_mode(df, position)
    else:
        render_profile_mode(df, position)

    render_glossary()
    render_methodology_note()


if __name__ == "__main__":
    main()
