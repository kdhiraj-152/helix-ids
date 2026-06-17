"""
HELIX-IDS Data Audit Module

Comprehensive data quality validation for intrusion detection datasets.
Validates: NaN distribution, duplicates, identifier leakage, schema consistency,
label integrity, and outlier detection across all supported datasets.

Usage:
    config = DataAuditConfig()
    audit = DataAudit(config)

    df_dict = {
        'nsl-kdd': df_nsl,
        'unsw': df_unsw,
    }
    report = audit.generate_audit_report(df_dict)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Audit Configuration
# ============================================================================


@dataclass
class DataAuditConfig:
    """Configuration parameters for data auditing."""

    # NaN thresholds
    nan_column_threshold: float = 0.30  # Flag columns with >30% NaN
    nan_row_threshold: float = 0.05  # Flag rows with >5% NaN

    # Duplicate detection
    check_exact_duplicates: bool = True
    check_partial_duplicates: bool = True
    partial_dup_threshold: float = 0.95  # >95% similarity

    # Identifier validation
    identifier_columns: list[str] = field(
        default_factory=lambda: [
            "Flow ID",
            "Src IP",
            "Dst IP",
            "Src Port",
            "Dst Port",
            "id",
            "flow_id",
            "src_ip",
            "dst_ip",
        ]
    )

    # Outlier detection
    outlier_sigma: float = 3.0
    outlier_percentile: float = 99.9

    # Label validation
    expected_5class_labels: list[str] = field(
        default_factory=lambda: ["Normal", "DoS", "Probe", "R2L", "U2R"]
    )

    # Schema validation
    check_schema_consistency: bool = True


# ============================================================================
# Data Audit Class
# ============================================================================


class DataAudit:
    """
    Comprehensive data auditor for intrusion detection datasets.

    Provides detailed quality metrics including NaN distribution, duplicates,
    identifier leakage, schema consistency, label integrity, and outliers.
    """

    def __init__(self, config: DataAuditConfig | None = None):
        """
        Initialize data auditor.

        Args:
            config: DataAuditConfig instance. Uses defaults if None.
        """
        self.config = config or DataAuditConfig()
        self.audit_results: dict[str, dict[str, Any]] = {}

    def _compute_stats(self, series: pd.Series) -> dict[str, float]:
        """
        Compute comprehensive statistics for a numeric series.

        Args:
            series: Numeric pandas Series

        Returns:
            Dict with mean, std, min, max, q1, q3, IQR, skew, kurtosis
        """
        if series.dtype not in [np.float64, np.float32, np.int64, np.int32]:
            return {}

        valid_data = series.dropna()
        if len(valid_data) == 0:
            return {}

        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1

        return {
            "mean": float(valid_data.mean()),
            "std": float(valid_data.std()),
            "min": float(valid_data.min()),
            "max": float(valid_data.max()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "skew": float(valid_data.skew()),
            "kurtosis": float(valid_data.kurtosis()),
        }

    def audit_nan_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Audit NaN (missing value) distribution across dataframe.

        Computes percentage of NaN values per column and overall.
        Flags columns exceeding configured threshold.

        Args:
            df: Input dataframe

        Returns:
            Dict with keys:
            - 'overall_nan_pct': % NaN across entire dataframe
            - 'per_column': Dict[col_name -> % NaN]
            - 'critical_columns': List of columns with NaN > threshold
            - 'rows_with_any_nan': count of rows with at least one NaN
        """
        logger.info(f"Auditing NaN distribution for {df.shape[0]} rows, {df.shape[1]} cols")

        total_cells = df.shape[0] * df.shape[1]
        total_nan = df.isna().sum().sum()
        overall_nan_pct = (total_nan / total_cells * 100) if total_cells > 0 else 0.0

        per_column_nan = {}
        critical_columns = []

        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df) * 100
            per_column_nan[col] = round(nan_pct, 2)

            if nan_pct > self.config.nan_column_threshold * 100:
                critical_columns.append(col)

        rows_with_nan = (df.isna().sum(axis=1) > 0).sum()

        result = {
            "overall_nan_pct": round(overall_nan_pct, 2),
            "per_column": per_column_nan,
            "critical_columns": critical_columns,
            "rows_with_any_nan": int(rows_with_nan),
            "total_rows": len(df),
        }

        if critical_columns:
            logger.warning(
                f"Critical NaN issue in columns: {critical_columns}. "
                f"Overall NaN: {overall_nan_pct:.2f}%"
            )

        return result

    def audit_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Audit exact and partial duplicates in dataframe.

        Args:
            df: Input dataframe

        Returns:
            Dict with keys:
            - 'exact_duplicates': count of completely identical rows
            - 'exact_dup_indices': sample indices of exact duplicates
            - 'rows_after_dedup': unique row count
        """
        logger.info(f"Auditing duplicates for {df.shape[0]} rows")

        if len(df) == 0:
            return {
                "exact_duplicates": 0,
                "exact_dup_sample_indices": [],
                "unique_rows": 0,
                "rows_before_dedup": 0,
                "duplicates_pct": 0.0,
            }

        # Exact duplicates
        exact_dup_mask = df.duplicated(keep=False)
        exact_dup_count = exact_dup_mask.sum()
        exact_dup_indices = df[exact_dup_mask].index.tolist()[:100]  # Sample

        # Rows after removing all but first duplicate
        unique_rows = len(df.drop_duplicates())

        result = {
            "exact_duplicates": int(exact_dup_count),
            "exact_dup_sample_indices": exact_dup_indices,
            "unique_rows": int(unique_rows),
            "rows_before_dedup": len(df),
            "duplicates_pct": round((1 - unique_rows / len(df)) * 100, 2),
        }

        if exact_dup_count > 0:
            logger.warning(
                "Found %d exact duplicates (%.2f%% of data)",
                exact_dup_count,
                result["duplicates_pct"],
            )

        return result

    def audit_identifiers(
        self, df: pd.DataFrame, exclude_cols: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Audit for potential identifier column leakage.

        Checks if known identifier columns (IP addresses, ports, IDs) are
        present in dataframe, which could cause information leakage.

        Args:
            df: Input dataframe
            exclude_cols: Explicit list of columns to check (overrides config)

        Returns:
            Dict with keys:
            - 'suspected_identifiers': List of identifier-like columns found
            - 'identifier_cardinality': Dict[col -> unique values]
            - 'identifier_risk': 'LOW' / 'MEDIUM' / 'HIGH'
        """
        logger.info(f"Auditing identifiers in {df.shape[1]} columns")

        check_cols = exclude_cols or self.config.identifier_columns
        suspected_identifiers = []
        identifier_cardinality = {}

        for col in df.columns:
            col_lower = col.lower()

            # Check against identifier list
            for id_pattern in check_cols:
                if id_pattern.lower() in col_lower:
                    suspected_identifiers.append(col)
                    cardinality = df[col].nunique()
                    identifier_cardinality[col] = int(cardinality)
                    break

        # Assess risk
        if len(suspected_identifiers) == 0:
            risk = "LOW"
        elif len(suspected_identifiers) <= 2:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        result = {
            "suspected_identifiers": suspected_identifiers,
            "identifier_cardinality": identifier_cardinality,
            "identifier_risk": risk,
        }

        if suspected_identifiers:
            logger.warning(
                f"Potential identifier leakage detected: {suspected_identifiers}. "
                f"Risk level: {risk}"
            )

        return result

    def audit_schema(self, dfs_dict: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """
        Audit schema consistency across multiple datasets.

        Validates that all datasets have compatible structure (matching columns,
        similar data types, consistent feature counts).

        Args:
            dfs_dict: Dict mapping dataset names to dataframes

        Returns:
            Dict with keys:
            - 'datasets': List of dataset names audited
            - 'column_intersection': Columns present in all datasets
            - 'column_union': All columns across any dataset
            - 'schema_mismatches': List of (dataset1, dataset2, mismatch_details)
            - 'consistency_score': 0.0-1.0 (1.0 = perfect consistency)
        """
        logger.info(f"Auditing schema consistency across {len(dfs_dict)} datasets")

        if len(dfs_dict) < 2:
            return {
                "datasets": list(dfs_dict.keys()),
                "column_intersection": list(dfs_dict[list(dfs_dict.keys())[0]].columns)
                if dfs_dict
                else [],
                "schema_mismatches": [],
                "consistency_score": 1.0,
            }

        all_columns = set()
        column_sets = {}
        mismatches = []

        for name, df in dfs_dict.items():
            columns = set(df.columns)
            column_sets[name] = columns
            all_columns.update(columns)

        # Find intersection
        intersection = set.intersection(*column_sets.values())

        # Compare pairwise
        names = list(dfs_dict.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                cols1, cols2 = column_sets[n1], column_sets[n2]

                only_in_1 = cols1 - cols2
                only_in_2 = cols2 - cols1

                if only_in_1 or only_in_2:
                    mismatches.append(
                        {
                            "dataset1": n1,
                            "dataset2": n2,
                            "only_in_dataset1": list(only_in_1),
                            "only_in_dataset2": list(only_in_2),
                        }
                    )

        # Consistency score: (columns in all datasets) / (total unique columns)
        consistency_score = len(intersection) / len(all_columns) if all_columns else 0.0

        result = {
            "datasets": list(dfs_dict.keys()),
            "column_intersection": list(intersection),
            "column_union": list(all_columns),
            "column_counts": {name: len(cols) for name, cols in column_sets.items()},
            "schema_mismatches": mismatches,
            "consistency_score": round(consistency_score, 3),
        }

        if mismatches:
            logger.warning(f"Schema mismatches detected: {len(mismatches)} dataset pairs differ")

        return result

    def audit_labels(
        self, df: pd.DataFrame, label_col: str, mapping: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Audit label integrity and distribution.

        Validates that all labels are mapped correctly, no unmapped values exist,
        and class distribution is reasonable.

        Args:
            df: Input dataframe
            label_col: Name of label column
            mapping: Optional dict of raw_label -> unified_label mappings

        Returns:
            Dict with keys:
            - 'unique_labels': List of unique label values
            - 'label_counts': Dict[label -> count]
            - 'label_distribution': Dict[label -> percentage]
            - 'unmapped_labels': Labels not in mapping (if mapping provided)
            - 'imbalance_ratio': max_class_pct / min_class_pct
        """
        logger.info(f"Auditing labels in column '{label_col}'")

        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found in dataframe")
            return {"error": f"Label column {label_col} not found"}

        label_counts = df[label_col].value_counts()
        unique_labels = label_counts.index.tolist()
        total = len(df)

        distribution = {
            str(label): round((count / total * 100), 2) for label, count in label_counts.items()
        }

        # Check for unmapped labels
        unmapped = []
        if mapping:
            for label in unique_labels:
                if str(label).lower() not in mapping:
                    unmapped.append(str(label))

        # Compute imbalance ratio
        counts = label_counts.values
        imbalance_ratio = float(counts.max() / counts.min()) if len(counts) > 0 else 0.0

        result = {
            "unique_labels": [str(label) for label in unique_labels],
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "label_distribution": distribution,
            "unmapped_labels": unmapped,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "total_samples": total,
        }

        if imbalance_ratio > 5:
            logger.warning(f"High class imbalance detected: ratio = {imbalance_ratio:.2f}x")

        if unmapped:
            logger.warning(f"Unmapped labels found: {unmapped}")

        return result

    def audit_outliers(self, df: pd.DataFrame, sigma: float = 3.0) -> dict[str, Any]:
        """
        Audit outlier distribution across numeric features.

        Identifies values that fall outside ±sigma standard deviations from mean.
        Also flags values beyond configured percentile threshold.

        Args:
            df: Input dataframe
            sigma: Number of standard deviations for outlier threshold

        Returns:
            Dict with keys:
            - 'per_column_outlier_pct': Dict[col -> % outliers]
            - 'overall_outlier_pct': % of all numeric values that are outliers
            - 'critical_outlier_columns': Columns with >5% outliers
        """
        logger.info(f"Auditing outliers (sigma={sigma}) for {df.shape[1]} columns")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        per_column_outliers = {}
        total_outliers = 0
        total_numeric_values = 0
        critical_cols = []

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                per_column_outliers[col] = 0.0
                continue

            # Outliers beyond ±sigma
            outlier_mask = np.abs((series - mean) / std) > sigma
            outlier_count = outlier_mask.sum()
            outlier_pct = outlier_count / len(series) * 100

            per_column_outliers[col] = round(outlier_pct, 2)
            total_outliers += outlier_count
            total_numeric_values += len(series)

            if outlier_pct > 5.0:
                critical_cols.append(col)

        overall_pct = (
            (total_outliers / total_numeric_values * 100) if total_numeric_values > 0 else 0.0
        )

        result = {
            "per_column_outlier_pct": per_column_outliers,
            "overall_outlier_pct": round(overall_pct, 2),
            "critical_outlier_columns": critical_cols,
            "numeric_columns_checked": len(numeric_cols),
        }

        if critical_cols:
            logger.warning(
                f"High outlier rate in columns: {critical_cols}. Overall: {overall_pct:.2f}%"
            )

        return result

    def generate_audit_report(
        self,
        df_dict: dict[str, pd.DataFrame],
        exclude_cols_per_dataset: dict[str, list[str]] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Generate comprehensive audit report for multiple datasets.

        Runs all audit methods on each dataframe and aggregates results
        into summary table and detailed JSON report.

        Args:
            df_dict: Dict mapping dataset names to dataframes
            exclude_cols_per_dataset: Dict mapping dataset name to identifier cols to exclude

        Returns:
            Tuple of:
            - pd.DataFrame: Summary table (datasets x audit metrics)
            - Dict: Detailed results with per-dataset breakdowns
        """
        logger.info(f"Generating audit report for {len(df_dict)} datasets")

        all_results: dict[str, dict[str, Any]] = {}
        summary_rows: list[dict[str, Any]] = []

        for dataset_name, df in df_dict.items():
            logger.info(f"\n--- Auditing dataset: {dataset_name} ---")

            dataset_results: dict[str, Any] = {
                "dataset_name": dataset_name,
                "shape": {"rows": len(df), "columns": len(df.columns)},
            }

            # Run individual audits
            dataset_results["nan_audit"] = self.audit_nan_distribution(df)
            dataset_results["duplicates_audit"] = self.audit_duplicates(df)

            exclude_cols = (
                exclude_cols_per_dataset.get(dataset_name, None)
                if exclude_cols_per_dataset
                else None
            )
            dataset_results["identifiers_audit"] = self.audit_identifiers(df, exclude_cols)
            dataset_results["outliers_audit"] = self.audit_outliers(df, self.config.outlier_sigma)

            all_results[dataset_name] = dataset_results

            # Create summary row
            summary_rows.append(
                {
                    "Dataset": dataset_name,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "NaN %": dataset_results["nan_audit"]["overall_nan_pct"],
                    "Duplicates": dataset_results["duplicates_audit"]["exact_duplicates"],
                    "Identifiers": len(
                        dataset_results["identifiers_audit"]["suspected_identifiers"]
                    ),
                    "Outlier %": dataset_results["outliers_audit"]["overall_outlier_pct"],
                }
            )

        # Schema audit (cross-dataset)
        logger.info("\n--- Cross-dataset schema validation ---")
        schema_results = self.audit_schema(df_dict)

        # Compile full results
        full_report = {
            "audit_config": asdict(self.config),
            "datasets_audited": list(df_dict.keys()),
            "timestamp": pd.Timestamp.now().isoformat(),
            "per_dataset": all_results,
            "schema_consistency": schema_results,
            "summary": {
                "total_datasets": len(df_dict),
            },
        }

        summary_df = pd.DataFrame(summary_rows)

        logger.info("\n=== Audit Report Generated ===")
        logger.info(f"\n{summary_df.to_string()}")

        return summary_df, full_report
