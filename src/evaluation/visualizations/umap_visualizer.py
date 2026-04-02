"""PCA + UMAP visualization for dual-encoder embeddings.

Implements Block 1 of the LiquidSearcher evaluation framework.
Uses two-stage dimensionality reduction:
1. PCA (50 components) for noise reduction - metrics computed here
2. UMAP (2D) for visualization only - NO metrics computed here
"""

import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not installed. UMAP visualization will not be available.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.evaluation.metrics.clustering import compute_all_clustering_metrics
from src.evaluation.utils.embedding_cache import EmbeddingCache
from src.evaluation.utils.feature_loader import FeatureLoader


class UMAPVisualizer:
    """PCA + UMAP visualization pipeline for dual-encoder embeddings."""

    PERIODS = {
        "covid_pre": ("2019-01-01", "2020-01-31"),
        "covid_crisis": ("2020-02-01", "2020-05-31"),
        "ratehike_pre": ("2021-01-01", "2021-12-31"),
        "ratehike_crisis": ("2022-01-01", "2022-10-31"),
        "normal_2019": ("2019-01-01", "2019-12-31"),
        "normal_2021": ("2021-01-01", "2021-12-31"),
    }

    def __init__(
        self,
        feature_loader: FeatureLoader,
        output_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        n_pca_components: int = 20,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
        checkpoint_path: Optional[str | Path] = None,
    ):
        """Initialize UMAP visualizer.

        Args:
            feature_loader: FeatureLoader for data access
            output_dir: Directory for output figures
            cache_dir: Directory for embedding cache
            n_pca_components: Number of PCA components
            umap_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            random_state: Random seed
            checkpoint_path: Path to model checkpoint (if None, uses placeholder)
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn required. Install: pip install umap-learn")

        self.feature_loader = feature_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cache_dir is None:
            cache_dir = self.output_dir / "cache"
        self.cache = EmbeddingCache(cache_dir)

        self.n_pca_components = n_pca_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state

        # Load model if checkpoint provided
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path:
            self._load_model(checkpoint_path)

        self.pca_model: Optional = None
        self.umap_model: Optional = None

    def _load_model(self, checkpoint_path: str | Path):
        """Load dual encoder model from checkpoint."""
        from src.models.dual_encoder import DualEncoder

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        hyperparams = checkpoint.get("hyper_parameters", {})
        self.model = DualEncoder(
            temporal_input_dim=hyperparams.get("temporal_input_dim", 13),
            tabular_continuous_dim=hyperparams.get("tabular_continuous_dim", 15),
            tabular_categorical_dims=hyperparams.get("tabular_categorical_dims"),
            tabular_embedding_dims=hyperparams.get("tabular_embedding_dims"),
            embedding_dim=hyperparams.get("embedding_dim", 128),
            temperature=hyperparams.get("temperature", 0.1),
        )

        # Handle Lightning checkpoint format
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def _extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract embedding columns from dataframe."""
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        return df[embedding_cols].values

    def _compute_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidity and market-cap tiers within period."""
        df = df.copy()

        if "market_cap" in df.columns:
            df["liquidity_tier"] = pd.qcut(
                df["market_cap"].rank(pct=True),
                q=4,
                labels=["Q4", "Q3", "Q2", "Q1"],
                duplicates="drop",
            )
        else:
            df["liquidity_tier"] = "Unknown"

        if "market_cap" in df.columns:
            df["market_cap_tier"] = pd.cut(
                df["market_cap"],
                bins=[0, 2e9, 10e9, 200e9, float("inf")],
                labels=["Micro", "Small", "Mid", "Large"],
            )
        else:
            df["market_cap_tier"] = "Unknown"

        return df

    def compute_embeddings(
        self,
        period_start: str,
        period_end: str,
        aggregation: Literal["end_period", "mean", "all_dates"] = "end_period",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute embeddings for specified period."""
        period_name = None
        for name, (start, end) in self.PERIODS.items():
            if start == period_start and end == period_end:
                period_name = name
                break

        model_id = "default" if self.model is None else getattr(self.model, "model_id", "default")

        if use_cache and period_name and self.cache.should_cache(period_name):
            cached = self.cache.load(period_name, model_id)
            if cached is not None:
                return cached

        print(f"Loading features for {period_start} to {period_end}...")
        features_df = self.feature_loader.load_period(period_start, period_end)

        if aggregation == "end_period":
            features_df = features_df.sort_values("date").groupby("symbol").last().reset_index()
            features_df = features_df.rename(columns={"symbol": "ticker"})
        elif aggregation == "mean":
            numeric_cols = features_df.select_dtypes(include=["float64", "int64"]).columns
            features_df = features_df.groupby("symbol")[numeric_cols].mean().reset_index()
            features_df = features_df.rename(columns={"symbol": "ticker"})
            features_df["date"] = pd.Timestamp(period_end)

        features_df = self._compute_tiers(features_df)

        if self.model is not None:
            print("Computing embeddings with dual encoder model...")
            embeddings = self._compute_model_embeddings(features_df)
        else:
            print("Computing embeddings (placeholder with sector structure)...")
            n_stocks = len(features_df)
            np.random.seed(self.random_state)
            sectors = (
                features_df.get("gsector", pd.Series([0] * n_stocks)).fillna(0).astype(int).values
            )
            n_sectors = max(sectors.max() + 1, 11)
            embeddings = np.zeros((n_stocks, 256))
            for i in range(n_stocks):
                sector_id = sectors[i] % n_sectors
                center = np.zeros(256)
                center[sector_id * 20 : (sector_id + 1) * 20] = 1.0
                embeddings[i] = center + np.random.randn(256) * 0.3

        for i in range(256):
            features_df[f"embedding_{i}"] = embeddings[:, i]

        base_cols = ["ticker", "date"]
        sector_cols = ["gsector", "ggroup"] if "gsector" in features_df.columns else []
        embedding_cols = [f"embedding_{i}" for i in range(256)]
        tier_cols = ["liquidity_tier", "market_cap_tier"]

        result_df = features_df[base_cols + sector_cols + embedding_cols + tier_cols].copy()

        if "gsector" in result_df.columns:
            result_df = result_df.rename(columns={"gsector": "sector"})

        if period_name and self.cache.should_cache(period_name):
            self.cache.save(result_df, period_name, model_id)

        return result_df

    def _compute_model_embeddings(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute embeddings using the trained dual encoder model."""
        from src.training.data_module import TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES

        n_stocks = len(features_df)
        embeddings = []

        temporal_cols = TEMPORAL_FEATURE_NAMES
        tabular_cols = TABULAR_CONTINUOUS_NAMES

        self.model.eval()
        with torch.no_grad():
            for idx in range(n_stocks):
                row = features_df.iloc[idx]

                # Prepare temporal features (60-day window would be ideal, using single day for now)
                temporal = torch.zeros(1, 60, 13)
                for i, col in enumerate(temporal_cols):
                    if col in row:
                        temporal[0, -1, i] = float(row[col]) if not pd.isna(row[col]) else 0.0

                # Prepare tabular features
                tabular_continuous = torch.zeros(1, 15)
                for i, col in enumerate(tabular_cols):
                    if col in row:
                        val = row[col]
                        tabular_continuous[0, i] = float(val) if not pd.isna(val) else 0.0

                # Prepare categorical (gsector, ggroup)
                # GICS codes are STRINGS: sectors '10'-'60', groups '1010'-'6020'
                # Map to 0-indexed: sector 10->0, 15->1, ..., 60->10 (11 total)
                # Map ggroup to 0-24 (25 total)
                categorical = torch.zeros(1, 2, dtype=torch.long)

                # Sector mapping: 10,15,20,25,30,35,40,45,50,55,60 -> 0,1,2,3,4,5,6,7,8,9,10
                SECTOR_MAP = {
                    "10": 0,
                    "15": 1,
                    "20": 2,
                    "25": 3,
                    "30": 4,
                    "35": 5,
                    "40": 6,
                    "45": 7,
                    "50": 8,
                    "55": 9,
                    "60": 10,
                }
                if "gsector" in row:
                    val = str(int(float(row["gsector"]))) if not pd.isna(row["gsector"]) else None
                    if val and val in SECTOR_MAP:
                        categorical[0, 0] = SECTOR_MAP[val]

                # Group mapping: create direct mapping for known groups
                if "ggroup" in row:
                    val = row["ggroup"]
                    if not pd.isna(val):
                        group_code = int(float(val))
                        # Map to 0-24 based on position in sorted unique values
                        group_idx = (group_code - 1010) // 100
                        categorical[0, 1] = max(0, min(24, group_idx))

                # Move to device
                temporal = temporal.to(self.device)
                tabular_continuous = tabular_continuous.to(self.device)
                categorical = categorical.to(self.device)

                # Get joint embedding
                emb = self.model.get_joint_embedding(temporal, tabular_continuous, categorical)
                embeddings.append(emb.cpu().numpy())

                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{n_stocks} stocks")

        return np.vstack(embeddings)

    def _compute_model_embeddings(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute embeddings using the trained dual encoder model."""
        from src.training.data_module import TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES

        n_stocks = len(features_df)
        embeddings = []

        temporal_cols = TEMPORAL_FEATURE_NAMES
        tabular_cols = TABULAR_CONTINUOUS_NAMES

        self.model.eval()
        with torch.no_grad():
            for idx in range(n_stocks):
                row = features_df.iloc[idx]

                # Prepare temporal features (60-day window would be ideal, using single day for now)
                temporal = torch.zeros(1, 60, 13)
                for i, col in enumerate(temporal_cols):
                    if col in row:
                        temporal[0, -1, i] = float(row[col]) if not pd.isna(row[col]) else 0.0

                # Prepare tabular features
                tabular_continuous = torch.zeros(1, 15)
                for i, col in enumerate(tabular_cols):
                    if col in row:
                        val = row[col]
                        tabular_continuous[0, i] = float(val) if not pd.isna(val) else 0.0

                # Prepare categorical (gsector, ggroup)
                # GICS codes are STRINGS: sectors '10'-'60', groups '1010'-'6020'
                # Map to 0-indexed: sector 10->0, 15->1, ..., 60->10 (11 total)
                # Map ggroup to 0-24 (25 total)
                categorical = torch.zeros(1, 2, dtype=torch.long)

                # Sector mapping: 10,15,20,25,30,35,40,45,50,55,60 -> 0,1,2,3,4,5,6,7,8,9,10
                SECTOR_MAP = {
                    "10": 0,
                    "15": 1,
                    "20": 2,
                    "25": 3,
                    "30": 4,
                    "35": 5,
                    "40": 6,
                    "45": 7,
                    "50": 8,
                    "55": 9,
                    "60": 10,
                }
                if "gsector" in row:
                    val = str(int(float(row["gsector"]))) if not pd.isna(row["gsector"]) else None
                    if val and val in SECTOR_MAP:
                        categorical[0, 0] = SECTOR_MAP[val]

                # Group mapping: create direct mapping for known groups
                if "ggroup" in row:
                    val = row["ggroup"]
                    if not pd.isna(val):
                        group_code = int(float(val))
                        # Map to 0-24 based on position in sorted unique values
                        group_idx = (group_code - 1010) // 100
                        categorical[0, 1] = max(0, min(24, group_idx))

                # Move to device
                temporal = temporal.to(self.device)
                tabular_continuous = tabular_continuous.to(self.device)
                categorical = categorical.to(self.device)

                # Get joint embedding
                emb = self.model.get_joint_embedding(temporal, tabular_continuous, categorical)
                embeddings.append(emb.cpu().numpy())

                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{n_stocks} stocks")

        return np.vstack(embeddings)

    def project_pca(
        self,
        embeddings: np.ndarray,
        fit_on: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply PCA for noise reduction."""
        from sklearn.decomposition import PCA

        fit_data = fit_on if fit_on is not None else embeddings

        self.pca_model = PCA(
            n_components=self.n_pca_components,
            whiten=False,
            random_state=self.random_state,
        )

        self.pca_model.fit(fit_data)

        var_explained = np.sum(self.pca_model.explained_variance_ratio_)
        print(f"PCA: {self.n_pca_components} components explain {var_explained:.2%} variance")

        return self.pca_model.transform(embeddings)

    def project_umap(
        self,
        pca_embeddings: np.ndarray,
        fit_mode: Literal["reference", "combined", "separate"] = "reference",
        reference_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply UMAP for 2D visualization ONLY."""
        if fit_mode == "reference":
            if reference_data is None:
                raise ValueError("reference_data required for 'reference' fit_mode")

            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            self.umap_model.fit(reference_data)
            return self.umap_model.transform(pca_embeddings)

        elif fit_mode == "combined":
            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            return self.umap_model.fit_transform(pca_embeddings)

        else:
            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            return self.umap_model.fit_transform(pca_embeddings)

    def generate_static_plot(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        title: str,
        filename: str,
    ) -> Path:
        """Generate publication-quality static PNG."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for static plots")

        column_map = {
            "sector": "sector" if "sector" in metadata.columns else "gsector",
            "liquidity": "liquidity_tier",
            "market_cap": "market_cap_tier",
        }
        color_column = column_map.get(color_by, color_by)

        if color_column not in metadata.columns:
            color_column = "sector" if "sector" in metadata.columns else "gsector"

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        if color_by == "sector":
            palette = "tab20"
        elif color_by == "liquidity":
            palette = "viridis"
        else:
            palette = "Set2"

        scatter = sns.scatterplot(
            data=metadata,
            x=projection[:, 0],
            y=projection[:, 1],
            hue=color_column,
            palette=palette,
            alpha=0.6,
            s=40,
            ax=ax,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(
            title=color_by.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved static plot: {output_path}")
        return output_path

    def generate_interactive_html(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        title: str,
        filename: str,
    ) -> Path:
        """Generate interactive Plotly HTML visualization."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for interactive plots")

        column_map = {
            "sector": "sector" if "sector" in metadata.columns else "gsector",
            "liquidity": "liquidity_tier",
            "market_cap": "market_cap_tier",
        }
        color_column = column_map.get(color_by, color_by)

        hover_data = ["ticker", "sector"] if "sector" in metadata.columns else ["ticker", "gsector"]
        hover_data = [h for h in hover_data if h in metadata.columns]

        fig = px.scatter(
            metadata,
            x=projection[:, 0],
            y=projection[:, 1],
            color=color_column,
            hover_data=hover_data,
            title=title,
            opacity=0.6,
        )

        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            showlegend=True,
            legend_title_text=color_by.replace("_", " ").title(),
        )

        output_path = self.output_dir / f"{filename}.html"
        fig.write_html(output_path)

        print(f"Saved interactive plot: {output_path}")
        return output_path

    def compute_clustering_metrics(
        self,
        embeddings_df: pd.DataFrame,
        pca_embeddings: np.ndarray,
    ) -> dict:
        """Compute clustering quality metrics in PCA space."""
        labels_dict = {}

        if "sector" in embeddings_df.columns:
            sector_labels, _ = pd.factorize(embeddings_df["sector"])
            labels_dict["sector"] = sector_labels
        elif "gsector" in embeddings_df.columns:
            labels_dict["sector"] = embeddings_df["gsector"].fillna(0).astype(int).values

        if "liquidity_tier" in embeddings_df.columns:
            liquidity_labels, _ = pd.factorize(embeddings_df["liquidity_tier"])
            labels_dict["liquidity"] = liquidity_labels

        if "market_cap_tier" in embeddings_df.columns:
            market_cap_labels, _ = pd.factorize(embeddings_df["market_cap_tier"])
            labels_dict["market_cap"] = market_cap_labels

        metrics = compute_all_clustering_metrics(pca_embeddings, labels_dict)

        for label_name in list(labels_dict.keys()):
            labels = labels_dict[label_name]
            shuffled_labels = np.random.RandomState(42).permutation(labels)
            random_score = compute_all_clustering_metrics(
                pca_embeddings,
                {label_name: shuffled_labels},
            )
            metrics[f"silhouette_{label_name}_random"] = random_score[f"silhouette_{label_name}"]

        return metrics

    def save_clustering_metrics(
        self,
        metrics: dict,
        period_name: str,
    ) -> Path:
        """Save clustering metrics to CSV."""
        output_path = self.output_dir / "clustering_metrics.csv"

        metrics_df = pd.DataFrame([metrics])
        metrics_df["period"] = period_name

        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, metrics_df], ignore_index=True)
            combined.to_csv(output_path, index=False)
        else:
            metrics_df.to_csv(output_path, index=False)

        print(f"Saved clustering metrics: {output_path}")
        return output_path

    def run_full_evaluation(self, period_key: str = "covid") -> dict:
        """Run full UMAP evaluation for a crisis period.

        Args:
            period_key: "covid" or "rate_hike"

        Returns:
            Dict with output paths and metrics
        """
        if period_key == "covid":
            pre_period = ("2019-01-01", "2020-01-31")
            crisis_period = ("2020-02-01", "2020-05-31")
        elif period_key == "rate_hike":
            pre_period = ("2021-01-01", "2021-12-31")
            crisis_period = ("2022-01-01", "2022-10-31")
        else:
            raise ValueError(f"Unknown period_key: {period_key}")

        print(f"\n{'=' * 60}")
        print(f"Running UMAP evaluation for {period_key.upper()}")
        print(f"{'=' * 60}\n")

        pre_df = self.compute_embeddings(pre_period[0], pre_period[1])
        pre_embed = self._extract_embeddings(pre_df)
        pre_pca = self.project_pca(pre_embed)

        print("\nGenerating static plots...")
        self.generate_static_plot(
            pre_pca,
            pre_df,
            "sector",
            f"UMAP by Sector - {period_key.title()} Pre-Crisis",
            f"umap_sector_{period_key}_pre",
        )
        self.generate_static_plot(
            pre_pca,
            pre_df,
            "liquidity",
            f"UMAP by Liquidity - {period_key.title()} Pre-Crisis",
            f"umap_liquidity_{period_key}_pre",
        )

        print("\nGenerating interactive plots...")
        self.generate_interactive_html(
            pre_pca,
            pre_df,
            "sector",
            f"UMAP by Sector - {period_key.title()} Pre-Crisis",
            f"umap_sector_{period_key}_pre",
        )

        print("\nComputing clustering metrics...")
        metrics = self.compute_clustering_metrics(pre_df, pre_pca)
        self.save_clustering_metrics(metrics, f"{period_key}_pre")

        print(f"\nMetrics for {period_key}_pre:")
        for key, value in metrics.items():
            if not key.endswith("_random"):
                print(f"  {key}: {value:.4f}")

        return {
            "period": period_key,
            "metrics": metrics,
            "n_stocks": len(pre_df),
        }
