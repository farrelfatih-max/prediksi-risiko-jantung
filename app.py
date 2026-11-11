"""
Aplikasi Streamlit: Prediksi Tingkat Risiko Penyakit Jantung
Berdasarkan notebook PROJEK_ML18.ipynb (dataset UCI Heart Disease - Cleveland).

Jalankan:  streamlit run app.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
RANDOM_STATE = 42
LABELS = ["Rendah", "Sedang", "Tinggi"]
COLORS = {"Rendah": "#2E8B57", "Sedang": "#E6A700", "Tinggi": "#C0392B"}
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

DATA_CANDIDATES = [
    "processed_cleveland.data", "processed.cleveland.data",
    "data/processed_cleveland.data", "data/processed.cleveland.data",
]

FEATURE_NAMES_ID = {
    "age": "Usia", "sex": "Jenis kelamin", "cp": "Tipe nyeri dada",
    "trestbps": "Tekanan darah istirahat", "chol": "Kolesterol",
    "fbs": "Gula darah puasa > 120", "restecg": "EKG istirahat",
    "thalach": "Detak jantung maksimum", "exang": "Angina akibat olahraga",
    "oldpeak": "ST depression (oldpeak)", "slope": "Kemiringan segmen ST",
    "ca": "Jumlah pembuluh darah utama", "thal": "Thalassemia",
}

st.set_page_config(page_title="Prediksi Risiko Jantung", page_icon="❤️", layout="wide")


# --------------------------------------------------------------------------
# Data & model (di-cache agar tidak dilatih ulang setiap interaksi)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in DATA_CANDIDATES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, names=COLS, na_values="?")
            df["risk"] = df["num"].map({0: 0, 1: 1, 2: 2, 3: 2, 4: 2})
            return df
    raise FileNotFoundError(
        "File data tidak ditemukan. Letakkan 'processed_cleveland.data' "
        "di folder yang sama dengan app.py."
    )


def make_prep(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                 ("oh", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num_steps), NUM_COLS),
                              ("cat", Pipeline(cat_steps), CAT_COLS)])


@st.cache_resource(show_spinner=False)
def train_all():
    """Melatih 3 model dengan pipeline yang sama seperti di notebook."""
    df = load_data()
    X, y = df[FEATURES], df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Logistic Regression": (
            Pipeline([("prep", make_prep(True)),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            Pipeline([("prep", make_prep(False)),
                      ("clf", RandomForestClassifier(class_weight="balanced",
                                                     random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None],
             "clf__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (
            Pipeline([("prep", make_prep(False)),
                      ("clf", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                            random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]}),
    }

    best, cv_scores, params = {}, {}, {}
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        fit_params = {}
        if name == "XGBoost":
            fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)
        gs.fit(X_train, y_train, **fit_params)
        best[name], cv_scores[name], params[name] = gs.best_estimator_, gs.best_score_, gs.best_params_

    rows, reports, cms = [], {}, {}
    for name, model in best.items():
        pred_te, pred_tr = model.predict(X_test), model.predict(X_train)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred_te),
            "Precision (macro)": precision_score(y_test, pred_te, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, pred_te, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, pred_te, average="macro", zero_division=0),
            "Recall kelas Tinggi": recall_score(y_test, pred_te, labels=[2], average=None,
                                                zero_division=0)[0],
            "F1 train": f1_score(y_train, pred_tr, average="macro"),
            "F1 CV": cv_scores[name],
        })
        reports[name] = pd.DataFrame(classification_report(
            y_test, pred_te, target_names=LABELS, zero_division=0, output_dict=True)).T
        cms[name] = confusion_matrix(y_test, pred_te, labels=[0, 1, 2])
    results = pd.DataFrame(rows).set_index("Model")

    # Model terbaik dipilih berdasarkan CV (bukan data uji), sama seperti notebook
    best_name = max(cv_scores, key=cv_scores.get)
    pi = permutation_importance(best[best_name], X_test, y_test, scoring="f1_macro",
                                n_repeats=30, random_state=RANDOM_STATE)
    importance = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()

    return {"models": best, "best_name": best_name, "results": results,
            "reports": reports, "cms": cms, "params": params,
            "importance": importance, "n_train": len(X_train), "n_test": len(X_test)}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("❤️ Prediksi Tingkat Risiko Penyakit Jantung")
st.caption("Proyek ML · Dataset UCI Heart Disease (Cleveland) · "
           "Kelas risiko: Rendah (num=0), Sedang (num=1), Tinggi (num=2–4)")

try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Menyiapkan model (hanya sekali saat pertama dibuka)…"):
    art = train_all()

best_name = art["best_name"]

tab_pred, tab_data, tab_model = st.tabs(["🩺 Prediksi", "📊 Eksplorasi Data", "🤖 Performa Model"])

# --------------------------------------------------------------------------
# Tab 1: Prediksi
# --------------------------------------------------------------------------
with tab_pred:
    model_choice = st.radio(
        "Model yang dipakai", list(art["models"].keys()),
        index=list(art["models"].keys()).index(best_name), horizontal=True,
        format_func=lambda n: f"{n} (terbaik)" if n == best_name else n)

    st.subheader("Data pasien")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Usia (tahun)", 20, 100, 58)
        sex = st.selectbox("Jenis kelamin", [1, 0],
                           format_func=lambda v: "Laki-laki" if v == 1 else "Perempuan")
        cp = st.selectbox("Tipe nyeri dada (cp)", [1, 2, 3, 4], index=3,
                          format_func=lambda v: {1: "1 · Angina tipikal", 2: "2 · Angina atipikal",
                                                 3: "3 · Nyeri non-angina",
                                                 4: "4 · Tanpa gejala (asymptomatic)"}[v])
        trestbps = st.number_input("Tekanan darah istirahat (mm Hg)", 80, 220, 150)
        chol = st.number_input("Kolesterol serum (mg/dl)", 100, 600, 270)
    with c2:
        fbs = st.selectbox("Gula darah puasa > 120 mg/dl", [0, 1],
                           format_func=lambda v: "Ya" if v == 1 else "Tidak")
        restecg = st.selectbox("Hasil EKG istirahat", [0, 1, 2], index=2,
                               format_func=lambda v: {0: "0 · Normal",
                                                      1: "1 · Kelainan gelombang ST-T",
                                                      2: "2 · Hipertrofi ventrikel kiri"}[v])
        thalach = st.number_input("Detak jantung maksimum (bpm)", 60, 220, 110)
        exang = st.selectbox("Angina akibat olahraga", [0, 1], index=1,
                             format_func=lambda v: "Ya" if v == 1 else "Tidak")
    with c3:
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 2.5, step=0.1)
        slope = st.selectbox("Kemiringan segmen ST puncak olahraga", [1, 2, 3], index=1,
                             format_func=lambda v: {1: "1 · Naik (upsloping)", 2: "2 · Datar (flat)",
                                                    3: "3 · Turun (downsloping)"}[v])
        ca = st.selectbox("Jumlah pembuluh darah utama (fluoroskopi)", [0, 1, 2, 3], index=2)
        thal = st.selectbox("Thalassemia", [3, 6, 7], index=2,
                            format_func=lambda v: {3: "3 · Normal", 6: "6 · Cacat tetap (fixed)",
                                                   7: "7 · Cacat reversibel"}[v])

    pasien = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal}])[FEATURES].astype(float)

    if st.button("Prediksi risiko", type="primary", width="stretch"):
        proba = art["models"][model_choice].predict_proba(pasien)[0]
        idx = int(np.argmax(proba))
        label = LABELS[idx]

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(f"#### Tingkat risiko: "
                        f"<span style='color:{COLORS[label]}'>**{label}**</span>",
                        unsafe_allow_html=True)
            st.metric("Keyakinan model", f"{proba[idx]:.1%}")
            if label == "Tinggi":
                st.error("Pola data pasien mirip kasus berisiko tinggi. Disarankan pemeriksaan "
                         "lanjutan oleh dokter.")
            elif label == "Sedang":
                st.warning("Pola data pasien mirip kasus berisiko sedang. Pemantauan dan "
                           "konsultasi medis dianjurkan.")
            else:
                st.success("Pola data pasien mirip kasus berisiko rendah.")
        with right:
            fig, ax = plt.subplots(figsize=(6, 2.6))
            bars = ax.barh(LABELS, proba * 100, color=[COLORS[l] for l in LABELS])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)")
            ax.invert_yaxis()
            for b, p in zip(bars, proba):
                ax.text(min(p * 100 + 1, 88), b.get_y() + b.get_height() / 2,
                        f"{p:.1%}", va="center", fontsize=9)
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

    st.info("⚠️ Aplikasi ini hanya untuk keperluan edukasi/proyek pembelajaran mesin dan "
            "**bukan pengganti diagnosis medis**.")

# --------------------------------------------------------------------------
# Tab 2: Eksplorasi data
# --------------------------------------------------------------------------
with tab_data:
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah pasien", len(df))
    m2.metric("Jumlah fitur", len(FEATURES))
    m3.metric("Nilai hilang", int(df[FEATURES].isna().sum().sum()))

    st.subheader("Distribusi tingkat risiko")
    dist = df["risk"].map(dict(enumerate(LABELS))).value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(dist.index, dist.values, color=[COLORS[l] for l in dist.index])
    for i, v in enumerate(dist.values):
        ax.text(i, v + 1, f"{v} ({v / len(df):.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Jumlah pasien")
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Hubungan fitur dengan tingkat risiko")
    feat = st.selectbox("Pilih fitur", FEATURES, format_func=lambda f: f"{f} · {FEATURE_NAMES_ID[f]}")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_df = df.assign(Risiko=df["risk"].map(dict(enumerate(LABELS))))
    if feat in NUM_COLS:
        sns.boxplot(data=plot_df, x="Risiko", y=feat, order=LABELS, palette=COLORS, ax=ax)
    else:
        sns.countplot(data=plot_df, x=feat, hue="Risiko", hue_order=LABELS, palette=COLORS, ax=ax)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Korelasi antar variabel")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[FEATURES + ["risk"]].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 7})
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat data mentah"):
        st.dataframe(df, width="stretch")

# --------------------------------------------------------------------------
# Tab 3: Performa model
# --------------------------------------------------------------------------
with tab_model:
    st.markdown(f"Model terbaik berdasarkan **CV F1-macro**: **{best_name}**  \n"
                f"Data latih: {art['n_train']} · Data uji: {art['n_test']} "
                f"(split 80/20 terstratifikasi, seed {RANDOM_STATE})")

    st.subheader("Perbandingan model (data uji)")
    st.dataframe(art["results"].round(4), width="stretch")

    metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    st.bar_chart(art["results"][metric_cols])

    st.subheader("Confusion matrix")
    cols = st.columns(3)
    for col, (name, cm) in zip(cols, art["cms"].items()):
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=10)
        plt.tight_layout()
        col.pyplot(fig)
        plt.close(fig)

    st.subheader("Laporan klasifikasi")
    sel = st.selectbox("Model", list(art["reports"].keys()),
                       index=list(art["reports"].keys()).index(best_name))
    st.dataframe(art["reports"][sel].round(3), width="stretch")
    st.caption(f"Hyperparameter terbaik: {art['params'][sel]}")

    st.subheader(f"Permutation importance ({best_name})")
    imp = art["importance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f"{i} · {FEATURE_NAMES_ID[i]}" for i in imp.index], imp.values, color="#C0392B")
    ax.set_xlabel("Penurunan F1 macro")
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Catatan: dataset kecil (±300 pasien), sehingga metrik pada data uji "
               "bervariasi cukup besar. Kelas 'Sedang' paling sulit dibedakan.")
