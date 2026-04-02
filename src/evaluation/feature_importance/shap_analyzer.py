"""SHAP analysis engine for dual-encoder similarity model."""

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.dual_encoder import DualEncoder
from src.training.data_module import TABULAR_CONTINUOUS_NAMES

warnings.filterwarnings("ignore")

CONTINUOUS_FEATURES = [
    "market_cap",
    "beta",
    "idiosyncratic_vol",
    "roe",
    "roa",
    "debt_to_equity",
    "price_to_book",
    "price_to_earnings",
    "operating_margin",
    "profit_margin",
    "dividend_yield",
    "revenue",
    "net_income",
    "total_assets",
    "cash",
]


class DualEncoderExplainer:
    """
    SHAP explainer for dual-encoder similarity model.

    Explains similarity scores as:
    similarity(q, c) = base_value + Σ φ_i

    where φ_i is the SHAP value for feature i.
    """

    def __init__(
        self,
        model: DualEncoder,
        background_data: pd.DataFrame,
        background_size: int = 100,
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained DualEncoder model
            background_data: DataFrame with tabular features for background samples
            background_size: Number of background samples (larger = more accurate, slower)
            device: Device for model computation
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)

        self.background_size = min(background_size, len(background_data))
        self.background_data = background_data.sample(
            n=self.background_size, random_state=42
        ).reset_index(drop=True)

        self.background_embeddings = self._compute_background_embeddings()

    def _compute_background_embeddings(self) -> torch.Tensor:
        """Pre-compute embeddings for background samples."""
        embeddings = []

        with torch.no_grad():
            for idx, row in self.background_data.iterrows():
                temporal, tabular, categorical = self._prepare_inputs(row)
                temporal = temporal.to(self.device)
                tabular = tabular.to(self.device)
                categorical = categorical.to(self.device)

                emb = self.model.get_joint_embedding(temporal, tabular, categorical)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    def _prepare_inputs(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare model inputs from a dataframe row."""
        temporal = torch.zeros(1, 60, 13)
        tabular = torch.zeros(1, 15)
        categorical = torch.zeros(1, 2, dtype=torch.long)

        for i, col in enumerate(
            [
                "z_close",
                "z_volume",
                "ma_ratio_5d",
                "ma_ratio_10d",
                "ma_ratio_20d",
                "realized_vol_20d",
                "realized_vol_60d",
                "mom_1m",
                "mom_3m",
                "mom_6m",
                "mom_12m",
                "mom_12_1m",
                "log_ret_cum",
            ]
        ):
            if col in row:
                val = row[col]
                if not pd.isna(val):
                    temporal[0, -1, i] = float(val)

        for i, col in enumerate(TABULAR_CONTINUOUS_NAMES):
            if col in row:
                val = row[col]
                if not pd.isna(val):
                    tabular[0, i] = float(val)

        if "gsector" in row:
            val = row["gsector"]
            if not pd.isna(val):
                sector_code = str(int(float(val)))
                sector_map = {
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
                if sector_code in sector_map:
                    categorical[0, 0] = sector_map[sector_code]

        if "ggroup" in row:
            val = row["ggroup"]
            if not pd.isna(val):
                ggroup_code = int(float(val))
                group_idx = (ggroup_code - 1010) // 100
                categorical[0, 1] = max(0, min(24, group_idx))

        return temporal, tabular, categorical

    def _create_similarity_wrapper(
        self,
        query_temporal: torch.Tensor,
        query_tabular: torch.Tensor,
        query_categorical: torch.Tensor,
    ) -> Callable:
        """
        Create a function that maps candidate tabular features → similarity score.

        This wrapper is passed to shap.DeepExplainer.

        Args:
            query_*: Pre-computed query inputs

        Returns:
            Function: candidate_tabular_features → similarity scores
        """
        # Extract query's tabular embedding for comparison (128-dim)
        # We compare tabular embeddings since we're explaining tabular feature effects
        with torch.no_grad():
            query_tabular_emb = self.model.tabular_encoder(
                query_tabular.to(self.device),
                query_categorical.to(self.device),
            )

        def similarity_fn(candidate_tabular_features: np.ndarray) -> np.ndarray:
            """
            Compute similarity between query and candidates.

            Args:
                candidate_tabular_features: (batch, 15) array of tabular features

            Returns:
                (batch,) array of similarity scores
            """
            batch_size = candidate_tabular_features.shape[0]

            candidate_tabular = torch.from_numpy(candidate_tabular_features).float()
            candidate_categorical = torch.zeros(batch_size, 2, dtype=torch.long)

            similarities = []
            with torch.no_grad():
                for i in range(batch_size):
                    tab = candidate_tabular[i : i + 1].to(self.device)
                    cat = candidate_categorical[i : i + 1].to(self.device)

                    candidate_tab_emb = self.model.tabular_encoder(tab, cat)
                    # Compare tabular embeddings (both 128-dim)
                    sim = F.cosine_similarity(query_tabular_emb, candidate_tab_emb, dim=-1)
                    similarities.append(sim.cpu().numpy())

            return np.array(similarities)

        return similarity_fn

    def explain_similarity(
        self,
        query_row: pd.Series,
        candidate_rows: pd.DataFrame,
        n_candidates: int = 50,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for similarity between query and candidates.

        Args:
            query_row: Query stock features
            candidate_rows: Candidate stock features
            n_candidates: Number of top candidates to explain

        Returns:
            DataFrame with columns:
            - candidate_ticker
            - similarity_score
            - shap_<feature_name> for each feature
            - residual (should be ~0)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Please install shap: pip install shap")

        query_temporal, query_tabular, query_categorical = self._prepare_inputs(query_row)

        candidate_temporals = []
        candidate_tabulars = []
        candidate_categoricals = []

        for idx, row in candidate_rows.iterrows():
            temp, tab, cat = self._prepare_inputs(row)
            candidate_temporals.append(temp)
            candidate_tabulars.append(tab)
            candidate_categoricals.append(cat)

        candidate_tabular_tensor = torch.cat(candidate_tabulars, dim=0)

        with torch.no_grad():
            # Use tabular embeddings for consistency with SHAP analysis
            query_tabular_emb = self.model.tabular_encoder(
                query_tabular.to(self.device),
                query_categorical.to(self.device),
            )

            candidate_embs = []
            for i in range(len(candidate_rows)):
                tab = candidate_tabular_tensor[i : i + 1].to(self.device)
                cat = candidate_categoricals[i].to(self.device)
                emb = self.model.tabular_encoder(tab, cat)
                candidate_embs.append(emb)

            candidate_embs = torch.cat(candidate_embs, dim=0)
            similarities = (
                F.cosine_similarity(query_tabular_emb, candidate_embs, dim=-1).cpu().numpy()
            )

        top_indices = np.argsort(similarities)[::-1][:n_candidates]
        top_candidates = candidate_rows.iloc[top_indices].reset_index(drop=True)
        top_similarities = similarities[top_indices]

        background_features = self.background_data[CONTINUOUS_FEATURES].values
        candidate_features = top_candidates[CONTINUOUS_FEATURES].values

        similarity_fn = self._create_similarity_wrapper(
            query_temporal, query_tabular, query_categorical
        )

        # Use KernelExplainer for model-agnostic SHAP (works with custom functions)
        # DeepExplainer requires TensorFlow and PyTorch model objects, not functions
        explainer = shap.KernelExplainer(
            model=similarity_fn,
            data=background_features,
        )

        shap_values = explainer.shap_values(candidate_features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Handle base value extraction (shape varies by explainer type)
        expected_val = explainer.expected_value
        if isinstance(expected_val, np.ndarray):
            if expected_val.ndim == 0:
                base_value = float(expected_val.item())
            else:
                base_value = float(expected_val.flat[0])
        else:
            base_value = float(expected_val)

        results = []
        for i in range(len(top_candidates)):
            row_data = {
                "query_ticker": query_row.get("symbol", "UNKNOWN"),
                "query_date": query_row.get("date", pd.NaT),
                "candidate_ticker": top_candidates.iloc[i].get("symbol", "UNKNOWN"),
                "similarity_score": float(top_similarities[i]),
                "base_value": base_value,
            }

            for j, feature in enumerate(CONTINUOUS_FEATURES):
                shap_val = shap_values[i, j]
                if isinstance(shap_val, np.ndarray):
                    shap_val = shap_val.item()
                row_data[f"shap_{feature}"] = float(shap_val)

            shap_sum = sum(row_data[f"shap_{feature}"] for feature in CONTINUOUS_FEATURES)
            predicted = base_value + shap_sum
            actual = float(top_similarities[i])
            row_data["residual"] = abs(predicted - actual)

            results.append(row_data)

        return pd.DataFrame(results)

    def explain_batch(
        self,
        queries_df: pd.DataFrame,
        all_stocks_df: pd.DataFrame,
        n_candidates_per_query: int = 50,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Run SHAP analysis for multiple queries.

        Args:
            queries_df: Query stocks to analyze
            all_stocks_df: All candidate stocks
            n_candidates_per_query: How many neighbors to explain
            output_dir: Directory to save per-query results

        Returns:
            Dictionary with aggregated statistics
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        all_shap_results = []
        feature_importance_rows = []

        for query_idx, query_row in queries_df.iterrows():
            query_ticker = query_row.get("symbol", "UNKNOWN")
            print(f"Processing query: {query_ticker}")

            try:
                shap_df = self.explain_similarity(
                    query_row=query_row,
                    candidate_rows=all_stocks_df,
                    n_candidates=n_candidates_per_query,
                )

                all_shap_results.append(shap_df)

                for feature in CONTINUOUS_FEATURES:
                    shap_col = f"shap_{feature}"
                    if shap_col in shap_df.columns:
                        feature_importance_rows.append(
                            {
                                "feature": feature,
                                "mean_abs_shap": shap_df[shap_col].abs().mean(),
                                "std_shap": shap_df[shap_col].std(),
                                "mean_shap": shap_df[shap_col].mean(),
                            }
                        )

                if output_dir:
                    query_date = query_row.get("date", pd.Timestamp.now())
                    if isinstance(query_date, pd.Timestamp):
                        date_str = query_date.strftime("%Y-%m-%d")
                    else:
                        date_str = "unknown"

                    output_path = output_dir / f"SHAP_{query_ticker}_{date_str}.parquet"
                    shap_df.to_parquet(output_path, index=False)
                    print(f"  Saved: {output_path}")

            except Exception as e:
                print(f"  Error processing {query_ticker}: {e}")
                continue

        if not all_shap_results:
            raise ValueError("No SHAP results generated")

        combined_df = pd.concat(all_shap_results, ignore_index=True)

        importance_df = pd.DataFrame(feature_importance_rows)
        importance_df["pct_importance"] = (
            importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum() * 100
        )
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

        residual_stats = {
            "mean_residual": combined_df["residual"].mean(),
            "max_residual": combined_df["residual"].max(),
            "pct_pass_validation": (combined_df["residual"] < 0.01).mean() * 100,
        }

        if residual_stats["mean_residual"] >= 0.01:
            warnings.warn(
                f"Additivity check failed: mean residual = {residual_stats['mean_residual']:.4f}"
            )

        return {
            "global_importance": importance_df,
            "per_query_results": combined_df,
            "residual_stats": residual_stats,
            "feature_names": CONTINUOUS_FEATURES,
            "shap_matrix": combined_df[[f"shap_{f}" for f in CONTINUOUS_FEATURES]].values,
        }
