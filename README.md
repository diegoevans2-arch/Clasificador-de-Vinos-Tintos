# 🍷 Clasificador de Vinos Tintos — Solemne 2

Aplicación Streamlit para publicar el modelo **Random Forest** desarrollado en la Solemne 1 del Taller de Aplicaciones. Permite visualizar el rendimiento del clasificador y probarlo ingresando datos por pantalla.

## 📦 Contenido

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Aplicación Streamlit (3 secciones) |
| `modelo_rf.pkl` | Random Forest entrenado (Exp. 3 — 16 features) |
| `metadata.pkl` | Métricas, importancias, rangos y reporte |
| `requirements.txt` | Dependencias |

## 🧪 Modelo

- **Algoritmo:** Random Forest (`n_estimators=100`, `random_state=42`)
- **Features:** 11 fisicoquímicas + 5 derivados (feature engineering del Experimento 3)
- **Target:** `quality_cat` (Malo / Neutro / Bueno)
- **Balanceo:** SMOTE manual sobre el train set
- **Split:** 80/20 estratificado, `random_state=42`

## 🖥 Secciones de la app

1. **📊 Resumen del Modelo** — métricas, matriz de confusión, reporte por clase, importancia de features
2. **🔮 Predicción Interactiva** — formulario con 11 inputs, calcula automáticamente los 5 derivados y predice la clase
3. **📈 Dataset** — distribución de clases y rangos por variable

## 🚀 Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁ Publicación en Streamlit Cloud

1. Sube este repositorio a GitHub (público).
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu GitHub.
3. New app → selecciona el repo → archivo principal: `app.py`.
4. Deploy → la URL pública estará lista en ~2 minutos.

## 📚 Dataset

Wine Quality — UCI Machine Learning Repository (Cortez et al., 2009). Subconjunto de vinos tintos: 1.599 registros, 11 variables fisicoquímicas + `quality`.
