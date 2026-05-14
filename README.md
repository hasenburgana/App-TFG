# Scout Profiles Lab

App web en Streamlit para explorar perfiles tácticos de jugadoras a partir del CSV exportado desde el notebook de clustering.

## Ejecutar en local

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar con GitHub en Streamlit Community Cloud

1. Sube este proyecto a GitHub.
2. Asegurate de incluir `data/perfiles_finales.csv` si los datos pueden ser publicos.
3. En Streamlit Community Cloud crea una app nueva.
4. Selecciona el repositorio, rama y `app.py` como archivo principal.
5. Deploy.

La app espera encontrar el archivo en `data/perfiles_finales.csv`.

## Exportar datos desde Colab

Cuando decidas el metodo final:

```python
metodo_final = resultados_pam_sinpca      # cambia por tu metodo elegido
col_cluster_final = "cluster"             # cluster, cluster_pca, cluster_hier o cluster_fa

dfs_final = []
for pos, res in metodo_final.items():
    df_pos = res["df"].copy()
    df_pos["grupo_pos_clustering"] = pos
    df_pos["cluster_final"] = df_pos[col_cluster_final]
    dfs_final.append(df_pos)

df_final = pd.concat(dfs_final, ignore_index=True)
df_final.to_csv("perfiles_finales.csv", index=False, encoding="utf-8-sig")
```

Coloca el archivo en:

```text
data/perfiles_finales.csv
```

## Funcionalidades

- Buscador de jugadoras por nombre o equipo.
- Radar interactivo p05-p95: jugadora vs perfil típico del cluster.
- Interpretación automática de perfiles por cluster.
- Glosario integrado con explicación del funcionamiento de la herramienta.
- Fortalezas y debilidades relativas del cluster.
- Jugadoras similares dentro de la misma posicion.
- Perfil supervisado: el usuario elige metricas y pesos, y se genera un ranking de encaje.

## Nota metodologica

Una silueta entre 0.15 y 0.25 no invalida automaticamente el TFG si los clusters tienen lectura futbolistica. En datos deportivos los perfiles suelen ser continuos y se solapan. Conviene reforzar la validacion con interpretabilidad: medias por cluster, medoides, estabilidad por bootstrap, separacion visual PCA/UMAP y validacion cualitativa de jugadoras conocidas.
