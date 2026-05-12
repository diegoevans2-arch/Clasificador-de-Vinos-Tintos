"""
Solemne 2 - Taller de Aplicaciones
Visualización y publicación del clasificador Random Forest (Solemne 1)
Dataset: Wine Quality - UCI (vinos tintos)
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Clasificador de Vinos Tintos",
    page_icon="🍷",
    layout="wide"
)


# ============================================================
# CARGA DE ARTEFACTOS (cacheada)
# ============================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_rf.pkl')
    metadata = joblib.load('metadata.pkl')
    return modelo, metadata

modelo, meta = cargar_modelo()


# ============================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================
st.sidebar.title("🍷 Clasificador de Vinos")
st.sidebar.markdown("**Solemne 2 — Taller de Aplicaciones**")
st.sidebar.markdown("---")

seccion = st.sidebar.radio(
    "Navegación",
    ["📊 Resumen del Modelo", "🔮 Predicción Interactiva", "📈 Dataset"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Modelo: Random Forest (Exp. 3)\n\n"
    "16 features (11 originales + 5 derivados)\n\n"
    "Entrenado con SMOTE manual sobre el train set."
)


# ============================================================
# SECCIÓN 1 - RESUMEN DEL MODELO
# ============================================================
if seccion == "📊 Resumen del Modelo":
    st.title("📊 Resumen del Clasificador")
    st.markdown(
        "Modelo **Random Forest** entrenado para clasificar la calidad de "
        "vinos tintos en 3 categorías: **Malo**, **Neutro** y **Bueno**."
    )

    # --- Métricas globales ---
    st.subheader("Métricas globales (test set)")
    m = meta['metricas']
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.4f}")
    c2.metric("F1 weighted", f"{m['f1_weighted']:.4f}")
    c3.metric("F1 macro", f"{m['f1_macro']:.4f}")

    st.markdown("---")

    # --- Matriz de confusión + Reporte por clase ---
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Matriz de Confusión")
        cm = np.array(meta['confusion_matrix'])
        labels = ['Malo', 'Neutro', 'Bueno']
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

    with col_b:
        st.subheader("Reporte por clase")
        report = meta['classification_report']
        df_report = pd.DataFrame({
            'Precision': [report[c]['precision'] for c in ['malo','neutro','bueno']],
            'Recall':    [report[c]['recall']    for c in ['malo','neutro','bueno']],
            'F1-score':  [report[c]['f1-score']  for c in ['malo','neutro','bueno']],
            'Support':   [int(report[c]['support']) for c in ['malo','neutro','bueno']]
        }, index=['Malo', 'Neutro', 'Bueno'])
        st.dataframe(df_report.round(4), use_container_width=True)

    st.markdown("---")

    # --- Importancia de features ---
    st.subheader("Importancia de Features")
    imp = pd.Series(meta['importancias']).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    colores = ['#d62728' if 'ratio' in f or 'acidez_total' in f or 'indice' in f
               else '#1f77b4' for f in imp.index]
    ax2.barh(imp.index, imp.values, color=colores, edgecolor='white')
    ax2.set_xlabel("Importancia")
    ax2.set_title("Features ordenados por importancia\n"
                  "(rojo = derivados del feature engineering)")
    plt.tight_layout()
    st.pyplot(fig2)


# ============================================================
# SECCIÓN 2 - PREDICCIÓN INTERACTIVA
# ============================================================
elif seccion == "🔮 Predicción Interactiva":
    st.title("🔮 Clasificar un vino")
    st.markdown(
        "Ingresa las **11 características fisicoquímicas** del vino. "
        "Los **5 atributos derivados** se calculan automáticamente."
    )

    features_input = meta['features_input_usuario']
    ranges = meta['ranges']

    st.subheader("Características fisicoquímicas")
    cols = st.columns(3)
    valores = {}
    for i, feat in enumerate(features_input):
        mn, mx, mean = ranges[feat]
        with cols[i % 3]:
            valores[feat] = st.number_input(
                label=feat,
                min_value=float(mn),
                max_value=float(mx),
                value=float(mean),
                step=(mx-mn)/100,
                format="%.4f",
                help=f"Rango dataset: [{mn:.3f}, {mx:.3f}]"
            )

    st.markdown("---")

    if st.button("🎯 Clasificar vino", type="primary", use_container_width=True):
        # Calcular features derivados (mismas fórmulas del notebook)
        v = valores
        v['ratio_so2']              = v['free sulfur dioxide'] / (v['total sulfur dioxide'] + 1e-9)
        v['ratio_acidez']           = v['fixed acidity'] / (v['volatile acidity'] + 1e-9)
        v['ratio_alcohol_densidad'] = v['alcohol'] / v['density']
        v['acidez_total']           = v['fixed acidity'] + v['volatile acidity'] + v['citric acid']
        v['indice_conservacion']    = v['sulphates'] * v['free sulfur dioxide']

        # Armar DataFrame en el orden correcto
        X_new = pd.DataFrame([[v[f] for f in meta['features_originales']]],
                             columns=meta['features_originales'])

        # Predecir
        pred = int(modelo.predict(X_new)[0])
        proba = modelo.predict_proba(X_new)[0]
        clase = meta['clases'][pred]

        # --- Resultado ---
        st.subheader("Resultado")
        emoji = {'Malo': '🔴', 'Neutro': '🟡', 'Bueno': '🟢'}[clase]
        st.markdown(f"### {emoji} Clase predicha: **{clase}**")
        st.markdown(f"Confianza: **{proba[pred]*100:.1f}%**")

        # --- Probabilidades por clase ---
        st.subheader("Probabilidades por clase")
        df_proba = pd.DataFrame({
            'Clase': ['Malo', 'Neutro', 'Bueno'],
            'Probabilidad': proba
        })
        fig3, ax3 = plt.subplots(figsize=(8, 2.5))
        colores_p = ['#d62728', '#ff7f0e', '#2ca02c']
        ax3.barh(df_proba['Clase'], df_proba['Probabilidad'],
                 color=colores_p, edgecolor='white')
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probabilidad")
        for i, p in enumerate(df_proba['Probabilidad']):
            ax3.text(p + 0.01, i, f"{p*100:.1f}%", va='center')
        plt.tight_layout()
        st.pyplot(fig3)

        # --- Features derivados calculados ---
        with st.expander("Ver features derivados calculados"):
            derivados = ['ratio_so2', 'ratio_acidez', 'ratio_alcohol_densidad',
                         'acidez_total', 'indice_conservacion']
            df_der = pd.DataFrame({
                'Feature': derivados,
                'Valor': [round(v[d], 4) for d in derivados]
            })
            st.dataframe(df_der, use_container_width=True, hide_index=True)


# ============================================================
# SECCIÓN 3 - DATASET
# ============================================================
elif seccion == "📈 Dataset":
    st.title("📈 Información del Dataset")

    st.markdown("""
**Dataset:** Wine Quality — UCI Machine Learning Repository (Cortez et al., 2009)
**Subconjunto utilizado:** vinos tintos (`winequality-red.csv`)
**Registros:** 1.599
**Variables fisicoquímicas:** 11
**Variable objetivo original:** `quality` (entero, 3 a 8)
**Variable objetivo del clasificador:** `quality_cat` (3 clases)

### Categorización de quality
| Categoría | Criterio        | Etiqueta |
|-----------|-----------------|----------|
| Malo      | quality ≤ 5     | 0        |
| Neutro    | quality = 6     | 1        |
| Bueno     | quality ≥ 7     | 2        |
""")

    st.markdown("---")
    st.subheader("Distribución de clases")
    dist = meta['distribucion_clases']
    df_dist = pd.DataFrame({
        'Clase':     ['Malo', 'Neutro', 'Bueno'],
        'Registros': [dist[0], dist[1], dist[2]]
    })
    df_dist['%'] = (df_dist['Registros'] / df_dist['Registros'].sum() * 100).round(2)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(df_dist, use_container_width=True, hide_index=True)
    with c2:
        fig4, ax4 = plt.subplots(figsize=(7, 3.5))
        colores_d = ['#d62728', '#ff7f0e', '#2ca02c']
        ax4.bar(df_dist['Clase'], df_dist['Registros'],
                color=colores_d, edgecolor='white')
        for i, n in enumerate(df_dist['Registros']):
            ax4.text(i, n+5, str(n), ha='center', fontsize=11)
        ax4.set_ylabel("Registros")
        ax4.set_title("Distribución de quality_cat")
        plt.tight_layout()
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("Rango de cada variable fisicoquímica")
    df_ranges = pd.DataFrame([
        {'Variable': f, 'Min': r[0], 'Max': r[1], 'Media': r[2]}
        for f, r in meta['ranges'].items()
        if f in meta['features_input_usuario']
    ]).round(4)
    st.dataframe(df_ranges, use_container_width=True, hide_index=True)
