#src/Charts/plot_features.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
import io


# ===== ЗАГРУЗКА =====
df = pd.read_csv("data/processed/my_traffic_features.csv")
print("Загружено строк:", len(df))
print("Колонки:", df.columns.tolist())

label_col = "label"

# ===== ВЫБОР ОСНОВНЫХ ПРИЗНАКОВ =====
scatter_x = "duration"
scatter_y = "bytes"
scatter3d_z = "pps"

for col in [scatter_x, scatter_y, scatter3d_z]:
    if col not in df.columns:
        raise ValueError(f"Признак {col} не найден в данных!")

# ====== ГИСТОГРАММА ======
plt.figure()
plt.hist(df[scatter_x], bins=50)
plt.title(f"Histogram: {scatter_x}")
plt.xlabel(scatter_x)
plt.ylabel("Count")
plt.show()

# ====== DIAGRAM SCATTER ======
plt.figure()
plt.scatter(df[scatter_x], df[scatter_y], alpha=0.3)
plt.title(f"Scatter: {scatter_x} vs {scatter_y}")
plt.xlabel(scatter_x)
plt.ylabel(scatter_y)
plt.show()

# ====== 3D SCATTER ======
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df[scatter_x], df[scatter_y], df[scatter3d_z], alpha=0.15)
ax.set_xlabel(scatter_x)
ax.set_ylabel(scatter_y)
ax.set_zlabel(scatter3d_z)
plt.title("3D Scatter")
plt.show()

# ====== BOXPLOT ======
plt.figure()
df[[scatter_x, scatter_y, scatter3d_z]].boxplot()
plt.title("Boxplot of selected features")
plt.show()

# ====== PCA 2D ======
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop(columns=[label_col]))

plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=3, alpha=0.3)
plt.title("PCA 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ====== PCA 3D ======
pca3 = PCA(n_components=3)
pca3_result = pca3.fit_transform(df.drop(columns=[label_col]))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pca3_result[:, 0], pca3_result[:, 1], pca3_result[:, 2], s=2, alpha=0.2)
ax.set_title("PCA 3D Projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()


