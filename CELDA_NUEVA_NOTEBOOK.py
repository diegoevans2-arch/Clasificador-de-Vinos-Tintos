# ============================================================
# SOLEMNE 2 — SERIALIZAR MODELO PARA APP STREAMLIT
# ============================================================
# Reproduce el RF del Experimento 3 (16 features) con el mismo split
# train/test y SMOTE manual del paso anterior, y guarda los artefactos
# necesarios para la app Streamlit en una carpeta `app/`.
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

# Carpeta destino
os.makedirs('app', exist_ok=True)

# --- Asegurar que el split y el modelo del Exp 3 estén disponibles ---
# Si ya ejecutaste la celda 24 del notebook, ya tienes rf_exp3, X_train_e3,
# X_test_e3, y_train_e3, y_test_e3. Lo siguiente es solo recalcular métricas
# y serializar.

y_pred_final = rf_exp3.predict(X_test_e3)

acc_final  = accuracy_score(y_test_e3, y_pred_final)
f1_w_final = f1_score(y_test_e3, y_pred_final, average='weighted', zero_division=0)
f1_m_final = f1_score(y_test_e3, y_pred_final, average='macro',    zero_division=0)
cm_final   = confusion_matrix(y_test_e3, y_pred_final)
report     = classification_report(y_test_e3, y_pred_final,
                                   target_names=['malo', 'neutro', 'bueno'],
                                   output_dict=True, zero_division=0)

# --- Rangos del dataset (para sliders/inputs de la app) ---
ranges = {
    f: (float(df_exp3[f].min()),
        float(df_exp3[f].max()),
        float(df_exp3[f].mean()))
    for f in features_exp3
}

# --- Metadata completa ---
metadata = {
    'features_originales': features_exp3,           # 16 features (orden modelo)
    'features_input_usuario': features,             # 11 originales (las del usuario)
    'clases': {0: 'Malo', 1: 'Neutro', 2: 'Bueno'},
    'metricas': {
        'accuracy':    float(acc_final),
        'f1_weighted': float(f1_w_final),
        'f1_macro':    float(f1_m_final)
    },
    'confusion_matrix': cm_final.tolist(),
    'classification_report': report,
    'importancias': dict(zip(features_exp3,
                              rf_exp3.feature_importances_.tolist())),
    'ranges': ranges,
    'distribucion_clases': df_tinto['quality_cat'].value_counts().sort_index().to_dict()
}

# --- Guardar ---
joblib.dump(rf_exp3,  'app/modelo_rf.pkl')
joblib.dump(metadata, 'app/metadata.pkl')

print("✅ Artefactos guardados en ./app/")
print(f"   - modelo_rf.pkl   ({os.path.getsize('app/modelo_rf.pkl')/1024:.1f} KB)")
print(f"   - metadata.pkl    ({os.path.getsize('app/metadata.pkl')/1024:.1f} KB)")
print()
print(f"Accuracy:    {acc_final:.4f}")
print(f"F1 weighted: {f1_w_final:.4f}")
print(f"F1 macro:    {f1_m_final:.4f}")
