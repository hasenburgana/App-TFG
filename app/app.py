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
