# src/visualization/feature_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import io

# reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# --------------------------
# HELPERS
# --------------------------
def save_fig_to_png_bytes(fig, dpi=150):
    """Сохранить matplotlib Figure в байтовый PNG (io.BytesIO)"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data(path="data/processed/my_traffic_features.csv"):
    return pd.read_csv(path)

df = load_data()
st.title("Визуализация сетевых признаков (Feature Explorer + PDF)")

st.write(f"Загружено строк: **{len(df):,}**")
st.write(f"Доступные признаки: **{len(df.columns)-1}**")

label_col = "label"
feature_cols = [c for c in df.columns if c != label_col]

# --------------------------
# SIDEBAR / UI
# --------------------------
st.sidebar.header("Параметры визуализации")
mode = st.sidebar.selectbox(
    "Тип анализа:",
    [
        "Histogram",
        "Scatter 2D",
        "Scatter 3D",
        "Boxplot",
        "PCA 2D",
        "PCA 3D",
        "Feature Importance + PDF Report"
    ]
)

feature_x = st.sidebar.selectbox("Признак X", feature_cols, index=0)
feature_y = st.sidebar.selectbox("Признак Y", feature_cols, index=1)
feature_z = st.sidebar.selectbox("Признак Z (3D)", feature_cols, index=2)

# --------------------------
# RENDER PLOTS (streamlit)
# --------------------------
if mode == "Histogram":
    st.subheader(f"Histogram — {feature_x}")
    fig, ax = plt.subplots()
    ax.hist(df[feature_x].dropna(), bins=50)
    ax.set_xlabel(feature_x)
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif mode == "Scatter 2D":
    st.subheader(f"Scatter Plot — {feature_x} vs {feature_y}")
    fig, ax = plt.subplots()
    ax.scatter(df[feature_x], df[feature_y], alpha=0.3, s=6)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    st.pyplot(fig)

elif mode == "Scatter 3D":
    st.subheader(f"Scatter 3D — {feature_x}, {feature_y}, {feature_z}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[feature_x], df[feature_y], df[feature_z], alpha=0.2, s=4)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(feature_z)
    st.pyplot(fig)

elif mode == "Boxplot":
    st.subheader(f"Boxplot — {feature_x}, {feature_y}, {feature_z}")
    fig, ax = plt.subplots(figsize=(8, 4))
    df[[feature_x, feature_y, feature_z]].boxplot(ax=ax)
    st.pyplot(fig)

elif mode == "PCA 2D":
    st.subheader("PCA 2D — проекция всего набора признаков")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feature_cols].fillna(0))
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(pca_result[:, 0], pca_result[:, 1], s=3, alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

elif mode == "PCA 3D":
    st.subheader("PCA 3D — проекция всего набора признаков")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feature_cols].fillna(0))
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], s=2, alpha=0.25)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    st.pyplot(fig)

# --------------------------
# FEATURE IMPORTANCE + PDF
# --------------------------
elif mode == "Feature Importance + PDF Report":
    st.subheader("Feature Importance (XGBoost)")

    # train XGBoost to get importances (on full dataset)
    st.info("Обучение XGBoost для вычисления важности признаков (несколько секунд)...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=4,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(df[feature_cols].fillna(0), df[label_col])

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": xgb.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    st.dataframe(imp_df.style.format({"Importance":"{:.6f}"}))

    # Plot importance (TOP 15)
    top_n = st.sidebar.slider("Top N features to show", min_value=5, max_value=min(30, len(imp_df)), value=15)
    imp_top = imp_df.head(top_n)

    fig_imp, ax_imp = plt.subplots(figsize=(8, max(4, top_n*0.25)))
    ax_imp.barh(imp_top["Feature"], imp_top["Importance"])
    ax_imp.invert_yaxis()
    ax_imp.set_title(f"Top {top_n} Important Features")
    st.pyplot(fig_imp)

    # PCA and Boxplots for inclusion in PDF
    # PCA 2D fig
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(df[feature_cols].fillna(0))
    fig_pca2, ax_pca2 = plt.subplots(figsize=(7,5))
    ax_pca2.scatter(pca_res[:,0], pca_res[:,1], s=3, alpha=0.4)
    ax_pca2.set_xlabel("PC1"); ax_pca2.set_ylabel("PC2")
    ax_pca2.set_title("PCA 2D Projection")

    # Boxplot top5 fig (or fewer if not enough)
    top5 = imp_df["Feature"].head(5).tolist()
    fig_box, ax_box = plt.subplots(figsize=(7,5))
    df[top5].boxplot(ax=ax_box)
    ax_box.set_title("Boxplot - Top 5 Features")

    # Convert figs to PNG bytes (to embed in PDF)
    png_importance = save_fig_to_png_bytes(fig_imp)
    png_pca2 = save_fig_to_png_bytes(fig_pca2)
    png_box = save_fig_to_png_bytes(fig_box)

    st.markdown("**Графики будут также включены в PDF-отчёт.**")

    # --------------------------
    # GENERATE PDF (embed PNGs)
    # --------------------------
    def generate_pdf_bytes(imp_df_table, png1, png2, png3):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=30, leftMargin=30,
                                topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Отчёт о важности сетевых признаков", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Всего строк: {len(df):,}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Table: top 20
        story.append(Paragraph("Топ 20 признаков по важности:", styles['Heading2']))
        story.append(Spacer(1, 6))
        table_data = [["Feature", "Importance"]]
        top20 = imp_df_table.head(20)
        for _, r in top20.iterrows():
            table_data.append([Paragraph(str(r["Feature"]), styles["BodyText"]), f"{float(r['Importance']):.6f}"])
        tbl = Table(table_data, colWidths=[300, 120])
        story.append(tbl)
        story.append(Spacer(1, 18))

        # Importance PNG
        story.append(Paragraph("График важности (Top {})".format(top_n), styles['Heading2']))
        story.append(Spacer(1,6))
        story.append(RLImage(png1, width=450, height= max(200, int(top_n*12))))
        story.append(Spacer(1, 18))

        # PCA PNG
        story.append(Paragraph("PCA 2D проекция", styles['Heading2']))
        story.append(Spacer(1,6))
        story.append(RLImage(png2, width=450, height=300))
        story.append(Spacer(1, 18))

        # Boxplot PNG
        story.append(Paragraph("Boxplot (Top 5)", styles['Heading2']))
        story.append(Spacer(1,6))
        story.append(RLImage(png3, width=450, height=300))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = generate_pdf_bytes(imp_df, png_importance, png_pca2, png_box)

    st.download_button(
        "Скачать PDF отчёт (таблица + графики)",
        data=pdf_bytes,
        file_name="feature_importance_report_full.pdf",
        mime="application/pdf"
    )
