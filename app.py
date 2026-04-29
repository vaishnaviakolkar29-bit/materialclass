"""
Advanced Material Property Predictor
=====================================
A fully rewritten, production-grade Streamlit application that:
  - Accepts any CSV dataset via file upload (no hardcoded paths)
  - Trains a RandomForest Classifier  (material type)
  - Trains a RandomForest Regressor   (tensile strength)
  - Uses 5-fold cross-validation for robust accuracy reporting
  - Displays feature importance, confusion matrix, actual vs predicted plots
  - Provides an interactive prediction panel with confidence bars

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Density(kg/m3)",
    "Tensile_Strength(MPa)",
    "Hardness(HB)",
    "Thermal_Conductivity(W/mK)",
    "Elastic_Modulus(GPa)",
]
TARGET_CLF  = "Class"
TARGET_REG  = "Tensile_Strength(MPa)"
REG_FEATURES = [f for f in FEATURE_COLS if f != TARGET_REG]

RF_CLF_PARAMS = dict(n_estimators=200, max_depth=None,
                     min_samples_split=3, random_state=42, n_jobs=-1)
RF_REG_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)

PALETTE = "#7c3aed"  # brand purple


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> list:
    """Return list of missing required columns (empty = OK)."""
    required = FEATURE_COLS + [TARGET_CLF]
    return [c for c in required if c not in df.columns]


@st.cache_data(show_spinner=False)
def load_data(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_classifier(df_hash: int, X: np.ndarray, y: np.ndarray, classes: tuple):
    """Train RandomForest classifier, return model + metrics."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(**RF_CLF_PARAMS)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    # y_te and y_pred are integer-encoded — use integer labels for confusion matrix
    int_labels = list(range(len(classes)))
    report = classification_report(y_te, y_pred, labels=int_labels,
                                   target_names=list(classes), output_dict=True)
    cm     = confusion_matrix(y_te, y_pred, labels=int_labels)  # ✅ fixed
    cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

    return clf, {"accuracy": acc, "cv_accuracy": cv_acc,
                 "report": report, "cm": cm,
                 "y_te": y_te, "y_pred": y_pred}


@st.cache_resource(show_spinner=False)
def train_regressor(df_hash: int, X: np.ndarray, y: np.ndarray):
    """Train RandomForest regressor, return model + metrics."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(**RF_REG_PARAMS)
    reg.fit(X_tr, y_tr)

    y_pred = reg.predict(X_te)
    mse    = mean_squared_error(y_te, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_te, y_pred)
    cv_r2  = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()

    return reg, {"mse": mse, "rmse": rmse, "r2": r2, "cv_r2": cv_r2,
                 "y_te": y_te, "y_pred": y_pred}


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    return fig, ax


def plot_confusion_matrix(cm: np.ndarray, classes: list) -> plt.Figure:
    fig, ax = _fig(6, 5)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Purples")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, title: str) -> plt.Figure:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = _fig(7, max(3, len(feature_names) * 0.55))
    bars = ax.barh(importances.index, importances.values, color=PALETTE, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_te, y_pred) -> plt.Figure:
    fig, ax = _fig(6, 5)
    ax.scatter(y_te, y_pred, alpha=0.6, color=PALETTE, edgecolors="none", s=40)
    lo = min(y_te.min(), y_pred.min())
    hi = max(y_te.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual Tensile Strength (MPa)", fontsize=10)
    ax.set_ylabel("Predicted (MPa)", fontsize=10)
    ax.set_title("Actual vs Predicted — Tensile Strength", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_residuals(y_te, y_pred) -> plt.Figure:
    residuals = y_te - y_pred
    fig, ax = _fig(6, 4)
    ax.scatter(y_pred, residuals, alpha=0.55, color="#2563eb", edgecolors="none", s=35)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted (MPa)", fontsize=10)
    ax.set_ylabel("Residual (MPa)", fontsize=10)
    ax.set_title("Residual Plot", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series) -> plt.Figure:
    counts = y.value_counts()
    fig, ax = _fig(6, 3.5)
    bars = ax.bar(counts.index, counts.values, color=PALETTE)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def render_metric_row(metrics: list):
    """Render a row of st.metric cards from a list of (label, value) tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def prediction_panel(clf, reg, le: LabelEncoder, classes: list):
    st.subheader("🔬 Interactive Prediction")
    st.caption("Adjust material properties and get an instant prediction.")

    c1, c2 = st.columns(2)
    with c1:
        density  = st.slider("Density (kg/m³)",            500,  20000,  5000, step=50)
        tensile  = st.slider("Tensile Strength (MPa)",        1,   2000,   300, step=5)
        hardness = st.slider("Hardness (HB)",                 1,   1000,   100, step=5)
    with c2:
        thermal  = st.slider("Thermal Conductivity (W/mK)", 0.1,  500.0,  50.0, step=0.5)
        elastic  = st.slider("Elastic Modulus (GPa)",         1,   1000,   100, step=5)

    inp_clf = pd.DataFrame(
        [[density, tensile, hardness, thermal, elastic]], columns=FEATURE_COLS
    )
    inp_reg = inp_clf[REG_FEATURES]

    # Predict — clf returns integer, decode to class name via LabelEncoder
    mat_enc      = clf.predict(inp_clf)[0]
    mat_type     = le.inverse_transform([mat_enc])[0]
    mat_proba    = clf.predict_proba(inp_clf)[0]
    tensile_pred = reg.predict(inp_reg)[0]

    st.divider()
    r1, r2 = st.columns(2)
    r1.success(f"**Predicted Material:** {mat_type}")
    r2.info(f"**Predicted Tensile Strength:** {tensile_pred:.1f} MPa")

    # Confidence bar chart — probabilities are ordered by clf.classes_ (integers)
    prob_df = (
        pd.DataFrame({"Probability": mat_proba}, index=classes)
        .sort_values("Probability", ascending=False)
    )
    fig, ax = _fig(6, 3)
    bars = ax.barh(prob_df.index[::-1], prob_df["Probability"].values[::-1], color=PALETTE)
    ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_title("Prediction Confidence", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Material Property Predictor",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚗️ Material Predictor")
        st.caption("Upload your materials dataset to begin.")
        uploaded = st.file_uploader("CSV Dataset", type=["csv"])

        st.divider()
        st.markdown("**Expected columns**")
        for col in FEATURE_COLS + [TARGET_CLF]:
            st.code(col, language=None)

        st.divider()
        st.markdown("**Model settings**")
        n_est = st.number_input("n_estimators", 50, 1000, 200, step=50)
        RF_CLF_PARAMS["n_estimators"] = n_est
        RF_REG_PARAMS["n_estimators"] = n_est

    # ── Gate on upload ────────────────────────────────────────────────────────
    if uploaded is None:
        st.info("👈 Upload a CSV dataset in the sidebar to get started.")
        st.stop()

    # ── Load & validate ───────────────────────────────────────────────────────
    raw_bytes = uploaded.read()
    df = load_data(raw_bytes)

    missing = validate_columns(df)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.dataframe(df.head())
        st.stop()

    df = df.dropna(subset=FEATURE_COLS + [TARGET_CLF]).reset_index(drop=True)

    # ── Encode labels ─────────────────────────────────────────────────────────
    classes = sorted(df[TARGET_CLF].unique().tolist())
    le      = LabelEncoder().fit(classes)
    y_enc   = le.transform(df[TARGET_CLF].values)

    X_clf = df[FEATURE_COLS].values
    X_reg = df[REG_FEATURES].values
    y_reg = df[TARGET_REG].values

    df_hash = hash(raw_bytes)  # stable cache key

    with st.spinner("Training models…"):
        # Pass classes as tuple so it's hashable for @st.cache_resource
        clf, clf_metrics = train_classifier(df_hash, X_clf, y_enc, tuple(classes))
        reg, reg_metrics = train_regressor(df_hash, X_reg, y_reg)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_predict, tab_clf, tab_reg, tab_data = st.tabs(
        ["🔬 Predict", "📊 Classifier", "📈 Regressor", "🗂 Dataset"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — Predict
    # ──────────────────────────────────────────────────────────────────────────
    with tab_predict:
        prediction_panel(clf, reg, le, classes)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Classifier
    # ──────────────────────────────────────────────────────────────────────────
    with tab_clf:
        st.subheader("Classification Performance")

        render_metric_row([
            ("Test Accuracy",        f"{clf_metrics['accuracy']*100:.2f}%"),
            ("CV Accuracy (5-fold)", f"{clf_metrics['cv_accuracy']*100:.2f}%"),
            ("Classes",              str(len(classes))),
            ("Training samples",     str(int(len(df) * 0.8))),
        ])

        st.divider()
        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(clf_metrics["cm"], classes)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_fi:
            st.markdown("**Feature Importance**")
            fig_fi = plot_feature_importance(clf, FEATURE_COLS, "Classifier Feature Importance")
            st.pyplot(fig_fi)
            plt.close(fig_fi)

        st.divider()
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(clf_metrics["report"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)

        st.divider()
        st.markdown("**Class Distribution**")
        fig_dist = plot_class_distribution(df[TARGET_CLF])
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — Regressor
    # ──────────────────────────────────────────────────────────────────────────
    with tab_reg:
        st.subheader("Regression Performance — Tensile Strength (MPa)")

        render_metric_row([
            ("RMSE",           f"{reg_metrics['rmse']:.2f} MPa"),
            ("MSE",            f"{reg_metrics['mse']:.2f}"),
            ("R² Score",       f"{reg_metrics['r2']:.4f}"),
            ("CV R² (5-fold)", f"{reg_metrics['cv_r2']:.4f}"),
        ])

        st.divider()
        col_avp, col_res = st.columns(2)

        with col_avp:
            fig_avp = plot_actual_vs_predicted(
                reg_metrics["y_te"], reg_metrics["y_pred"]
            )
            st.pyplot(fig_avp)
            plt.close(fig_avp)

        with col_res:
            fig_res = plot_residuals(
                reg_metrics["y_te"], reg_metrics["y_pred"]
            )
            st.pyplot(fig_res)
            plt.close(fig_res)

        st.divider()
        st.markdown("**Regressor Feature Importance**")
        fig_rfi = plot_feature_importance(reg, REG_FEATURES, "Regressor Feature Importance")
        st.pyplot(fig_rfi)
        plt.close(fig_rfi)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Dataset
    # ──────────────────────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Dataset Explorer")
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
        st.dataframe(df, use_container_width=True, height=420)

        st.divider()
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.divider()
        st.markdown("**Correlation Heatmap**")
        numeric_df = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            numeric_df.corr(), annot=True, fmt=".2f",
            cmap="Purples", ax=ax_corr, linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        ax_corr.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
