#src/Charts/scatter_plot.py
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter(df: pd.DataFrame, x: str, y: str, label_col: str = None, save_path: str = None):
    """
    Строит диаграмму рассеивания по двум выбранным фичам.

    :param df: DataFrame с фичами.
    :param x: Название фичи по оси X.
    :param y: Название фичи по оси Y.
    :param label_col: Колонка с метками (для раскраски).
    :param save_path: Путь для сохранения изображения.
    """

    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Колонки {x} или {y} отсутствуют в DataFrame")

    plt.figure(figsize=(8, 6))

    if label_col and label_col in df.columns:
        unique_labels = df[label_col].unique()
        for lbl in unique_labels:
            subset = df[df[label_col] == lbl]
            plt.scatter(subset[x], subset[y], label=str(lbl), alpha=0.6)
        plt.legend()
    else:
        plt.scatter(df[x], df[y], alpha=0.6)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter Plot: {x} vs {y}")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_hist(df: pd.DataFrame, column: str, bins: int = 30, save_path: str = None):
    """
    Строит гистограмму для выбранной фичи.
    """
    if column not in df.columns:
        raise ValueError(f"Колонка {column} отсутствует в DataFrame")

    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Histogram: {column}")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r"/data/processed/my_traffic_features.csv")
    plot_scatter(df, x="pps", y="bps", label_col="label", save_path="scatter_pps_bps.png")
    plot_hist(df, column="pps", bins=40, save_path="hist_pps.png")
