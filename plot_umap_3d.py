"""Interactive 3D UMAP plot coloured by KMeans cluster.

Reproduces the clustering pipeline from ``test_task.ipynb`` (negative-value filter,
ECOD outlier removal, StandardScaler, PCA, KMeans), embeds the PCA space used for
clustering into three UMAP components and writes the figure to HTML.
Hovering over a point shows its cluster and all original feature values.

Usage::

    uv run python plot_umap_3d.py [--output umap_3d_clusters.html] [--show]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyod.models.ecod import ECOD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

NON_NEGATIVE_COLUMNS = ["TotalInactiveDays", "ActivePassiveRatio"]
N_PLOT_COMPONENTS = 3


def load_data(path: Path) -> pd.DataFrame:
    """Load customer data and drop rows with invalid negative values.

    Parameters
    ----------
    path : Path
        Path to the semicolon-separated customer CSV.

    Returns
    -------
    pd.DataFrame
        Feature table without ``ClientId`` and without rows that have negative
        ``TotalInactiveDays`` or ``ActivePassiveRatio``.
    """
    data = pd.read_csv(path, sep=";", index_col=0)
    data = data.drop(columns=["ClientId"])
    return data[~(data[NON_NEGATIVE_COLUMNS] < 0).any(axis=1)]


def remove_outliers(data: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Drop multivariate outliers detected by ECOD.

    Parameters
    ----------
    data : pd.DataFrame
        Raw feature table.
    contamination : float
        Expected share of outliers, in ``(0, 0.5]``.

    Returns
    -------
    pd.DataFrame
        ``data`` without the rows flagged as outliers.
    """
    detector = ECOD(contamination=contamination)
    detector.fit(data)
    return data[detector.labels_ == 0]


def fit_clusters(
    data: pd.DataFrame,
    n_components: int | float,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the StandardScaler -> PCA -> KMeans pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned feature table.
    n_components : int or float
        ``PCA`` ``n_components``: number of components or explained variance share.
    n_clusters : int
        Number of KMeans clusters.
    n_init : int
        Number of KMeans initialisations.
    max_iter : int
        Maximum number of KMeans iterations.
    random_state : int
        Seed for KMeans, making cluster numbering reproducible.

    Returns
    -------
    components : np.ndarray of shape (n_samples, n_pca_components)
        All PCA components, i.e. the space used for clustering.
    labels : np.ndarray of shape (n_samples,)
        Cluster labels.
    """
    preprocessing = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_components))]
    )
    pipe = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "clusterer",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    labels = pipe.fit_predict(data)
    return preprocessing.transform(data), labels


def embed_umap(
    components: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int | None,
) -> np.ndarray:
    """Embed the clustering space into three UMAP components.

    Parameters
    ----------
    components : np.ndarray of shape (n_samples, n_pca_components)
        PCA components used for clustering.
    n_neighbors : int
        UMAP neighbourhood size.
    min_dist : float
        UMAP minimum distance between embedded points.
    random_state : int or None
        UMAP seed. A seed makes the embedding reproducible but forces UMAP to
        run single-threaded, which is several times slower.

    Returns
    -------
    np.ndarray of shape (n_samples, 3)
        UMAP embedding.
    """
    reducer = UMAP(
        n_components=N_PLOT_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(components)


def plot_umap_3d(
    data: pd.DataFrame,
    embedding: np.ndarray,
    labels: np.ndarray,
) -> go.Figure:
    """Build a 3D scatter of UMAP components coloured by cluster.

    Parameters
    ----------
    data : pd.DataFrame
        Feature table whose rows align with ``embedding`` and ``labels``;
        all its columns are shown on hover.
    embedding : np.ndarray of shape (n_samples, 3)
        UMAP embedding.
    labels : np.ndarray of shape (n_samples,)
        Cluster labels.

    Returns
    -------
    go.Figure
        Interactive plotly figure.
    """
    umap_columns = [f"UMAP{i + 1}" for i in range(N_PLOT_COMPONENTS)]
    plot_df = data.copy()
    plot_df[umap_columns] = embedding
    # String labels give a discrete palette and a clickable legend per cluster.
    plot_df["Cluster"] = labels.astype(str)

    hover_data = {column: ":,.4~f" for column in data.columns}
    hover_data.update({column: False for column in umap_columns})

    fig = px.scatter_3d(
        plot_df,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="Cluster",
        category_orders={"Cluster": [str(label) for label in np.unique(labels)]},
        hover_data=hover_data,
        title="UMAP embedding (3 components) by cluster",
    )
    fig.update_traces(marker={"size": 2})
    fig.update_layout(legend={"itemsizing": "constant"})
    return fig


def _n_components(value: str) -> int | float:
    """Parse PCA ``n_components`` as an int count or a float variance share."""
    return float(value) if "." in value else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("customer_data_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("umap_3d_clusters.html"))
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-components", type=_n_components, default=0.8)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    # Best 3D UMAP parameters from the Optuna search in test_task.ipynb
    parser.add_argument("--n-neighbors", type=int, default=180)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=None,
        help="Seed for a reproducible UMAP layout; makes UMAP single-threaded.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the figure in a browser."
    )
    args = parser.parse_args()

    data = remove_outliers(load_data(args.data), args.contamination)
    components, labels = fit_clusters(
        data,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    embedding = embed_umap(
        components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.umap_random_state,
    )
    fig = plot_umap_3d(data, embedding, labels)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved {len(data)} points in {args.n_clusters} clusters to {args.output}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
