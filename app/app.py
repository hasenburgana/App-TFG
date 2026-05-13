from __future__ import annotations

from pathlib import Path
from typing import Iterable
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
}


st.set_page_config(
    page_title="TFG Perfiladoras",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #f7f9fc;
            border-right: 1px solid #d8dee8;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 12px;
            background: #ffffff;
        }
        .profile-pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 8px;
            color: white;
            background: #2357c6;
            font-weight: 800;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(content: bytes | None) -> pd.DataFrame:
    if content is not None:
        return pd.read_csv(BytesIO(content))
    return pd.read_csv(DATA_PATH)


def load_data() -> pd.DataFrame | None:
    uploaded = st.sidebar.file_uploader(
        "CSV de perfiles",
        type=["csv"],
        help="Opcional. Si no subes archivo, se usa data/perfiles_finales.csv.",
    )

    if uploaded is None and not DATA_PATH.exists():
        st.error(
            "No encuentro `data/perfiles_finales.csv`. Sube el CSV en la barra lateral "
            "o colócalo en esa ruta dentro del repositorio."
        )
        return None

    content = uploaded.getvalue() if uploaded is not None else None
    df = read_csv_from_bytes(content)
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
    metrics = [
        col
        for col in numeric_cols
        if col not in METADATA_COLS and not col.startswith("cluster")
    ]
    return sorted(metrics)


def clean_metric_name(metric: str) -> str:
    return (
        metric.replace("_p90", "")
        .replace("pct_", "% ")
        .replace("_", " ")
    )


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


def cluster_percentiles(df_pos: pd.DataFrame, cluster: int, metrics: Iterable[str]) -> dict[str, float]:
    df_cluster = df_pos[df_pos[CLUSTER_COL] == cluster]
    result = {}
    for metric in metrics:
        mean_value = df_cluster[metric].mean()
        result[metric] = percentile(df_pos[metric], mean_value)
    return result


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
    top_metrics = top_discriminant_metrics(df_pos, metrics, 12)

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

    player_pct = player_percentiles(df_pos, player, top_metrics)
    cluster_pct = cluster_percentiles(df_pos, cluster, top_metrics)

    radar = go.Figure()
    theta = [clean_metric_name(metric) for metric in top_metrics]
    player_values = [player_pct[metric] for metric in top_metrics]
    cluster_values = [cluster_pct[metric] for metric in top_metrics]

    radar.add_trace(
        go.Scatterpolar(
            r=player_values + player_values[:1],
            theta=theta + theta[:1],
            fill="toself",
            name=str(player[PLAYER_COL]),
            line=dict(color="#2357c6", width=3),
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=cluster_values + cluster_values[:1],
            theta=theta + theta[:1],
            fill="toself",
            name=f"Media cluster {cluster}",
            line=dict(color="#0f8f8c", width=2),
        )
    )
    radar.update_layout(
        height=620,
        margin=dict(l=40, r=40, t=35, b=35),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        legend=dict(orientation="h"),
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Radar interactivo")
        st.plotly_chart(radar, use_container_width=True)

    with right:
        st.subheader("Lectura del cluster")
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


def render_profile_mode(df: pd.DataFrame, position: str) -> None:
    df_pos = df[df[POSITION_COL] == position].copy()
    metrics = metric_columns(df_pos, position)
    suggested = top_discriminant_metrics(df_pos, metrics, 18)

    st.subheader("Crear un perfil con métricas y pesos")
    st.caption("Peso positivo: busca valores altos. Peso negativo: busca valores bajos. Cero: no se usa.")

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


def main() -> None:
    inject_css()
    st.title("TFG Perfiladoras")
    st.caption("Explorador interactivo de perfiles de jugadoras con clustering y score supervisado.")

    df = load_data()
    if df is None:
        return

    positions = sorted(df[POSITION_COL].dropna().unique().tolist())
    position = st.sidebar.selectbox("Posición", positions)
    mode = st.sidebar.radio(
        "Forma de encontrar perfiles",
        ["A partir de una jugadora", "A partir de un perfil creado"],
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Jugadoras cargadas", len(df))
    st.sidebar.metric("Posiciones", len(positions))

    if mode == "A partir de una jugadora":
        render_player_mode(df, position)
    else:
        render_profile_mode(df, position)

    render_methodology_note()


if __name__ == "__main__":
    main()
