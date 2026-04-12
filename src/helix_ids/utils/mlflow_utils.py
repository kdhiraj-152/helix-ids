"""
MLflow Experiment Tracking Utilities for HELIX-IDS

Provides wrapper functions for MLflow integration with graceful degradation
when MLflow is not installed.

Functions:
    - init_mlflow: Initialize MLflow tracking
    - log_params: Log hyperparameters
    - log_metrics: Log metrics with optional step
    - log_model: Log and optionally register model
    - log_artifacts: Log artifacts from local path
    - start_run: Start a new MLflow run
    - end_run: End the current MLflow run
    - get_best_run: Query best run by metric
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional, Union, cast

# Check if MLflow is available
try:
    import mlflow  # type: ignore[import-not-found]
except ImportError:
    mlflow = None  # type: ignore[assignment]

MLFLOW_AVAILABLE = mlflow is not None

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


def init_mlflow(
    experiment_name: str,
    tracking_uri: str = "mlruns",
    artifact_location: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """
    Initialize MLflow tracking.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: URI for tracking (local path or remote server)
        artifact_location: Custom artifact storage location
        tags: Optional tags to set for the experiment

    Returns:
        Experiment ID if successful, None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Tracking disabled.")
        return None

    # Set tracking URI
    if not tracking_uri.startswith(("http://", "https://", "file://")):
        # Convert relative path to absolute for local storage
        tracking_uri = os.path.abspath(tracking_uri)
        tracking_uri = f"file://{tracking_uri}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location, tags=tags
        )
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return cast(str, experiment_id)


def log_params(params: dict[str, Any]) -> bool:
    """
    Log hyperparameters to MLflow.

    Args:
        params: Dictionary of parameter names and values

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        # MLflow requires string values, so convert non-strings
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value

        mlflow.log_params(sanitized)
        return True
    except Exception as e:
        logger.warning(f"Failed to log params: {e}")
        return False


def log_metrics(metrics: dict[str, Union[int, float]], step: Optional[int] = None) -> bool:
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (epoch)

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        # Filter out non-numeric values
        numeric_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
            and not (isinstance(v, float) and (v != v))  # exclude NaN
        }

        if step is not None:
            mlflow.log_metrics(numeric_metrics, step=step)
        else:
            mlflow.log_metrics(numeric_metrics)
        return True
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")
        return False


def log_model(
    model: Any,
    artifact_path: str,
    registered_name: Optional[str] = None,
    signature: Optional[Any] = None,
    input_example: Optional[Any] = None,
    pip_requirements: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Log a PyTorch model to MLflow.

    Args:
        model: PyTorch model to log
        artifact_path: Path within the run's artifact directory
        registered_name: If provided, register model with this name
        signature: Model signature for input/output schema
        input_example: Example input for documentation
        pip_requirements: Python dependencies

    Returns:
        Model URI if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        # Try PyTorch flavor first
        try:
            import mlflow.pytorch

            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )
        except Exception:
            # Fallback to generic log
            mlflow.log_artifact(artifact_path)
            return None

        model_uri = str(model_info.model_uri)
        logger.info(f"Model logged to: {model_uri}")
        return model_uri
    except Exception as e:
        logger.warning(f"Failed to log model: {e}")
        return None


def log_artifacts(local_path: str, artifact_path: Optional[str] = None) -> bool:
    """
    Log artifacts from a local path to MLflow.

    Args:
        local_path: Local path to file or directory
        artifact_path: Optional destination path in artifacts

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        if os.path.isdir(local_path):
            mlflow.log_artifacts(local_path, artifact_path)
        else:
            mlflow.log_artifact(local_path, artifact_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to log artifacts: {e}")
        return False


def log_figure(figure: Any, artifact_file: str) -> bool:
    """
    Log a matplotlib figure as an artifact.

    Args:
        figure: Matplotlib figure object
        artifact_file: Filename for the artifact (e.g., "confusion_matrix.png")

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.log_figure(figure, artifact_file)
        return True
    except Exception as e:
        logger.warning(f"Failed to log figure: {e}")
        return False


def log_dict(dictionary: dict, artifact_file: str) -> bool:
    """
    Log a dictionary as a JSON artifact.

    Args:
        dictionary: Dictionary to log
        artifact_file: Filename for the artifact (e.g., "config.json")

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.log_dict(dictionary, artifact_file)
        return True
    except Exception as e:
        logger.warning(f"Failed to log dict: {e}")
        return False


def start_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[dict[str, str]] = None,
    description: Optional[str] = None,
) -> Optional[Any]:
    """
    Start a new MLflow run.

    Args:
        run_name: Name for the run
        nested: Whether this is a nested run
        tags: Optional tags for the run
        description: Optional description for the run

    Returns:
        ActiveRun object if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        run = mlflow.start_run(run_name=run_name, nested=nested, tags=tags, description=description)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run
    except Exception as e:
        logger.warning(f"Failed to start run: {e}")
        return None


def end_run(status: str = "FINISHED") -> bool:
    """
    End the current MLflow run.

    Args:
        status: Final status ("FINISHED", "FAILED", "KILLED")

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.end_run(status=status)
        return True
    except Exception as e:
        logger.warning(f"Failed to end run: {e}")
        return False


@contextmanager
def mlflow_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    nested: bool = False,
    tags: Optional[dict[str, str]] = None,
):
    """
    Context manager for MLflow runs with automatic cleanup.

    Usage:
        with mlflow_run("my_run", experiment_name="my_experiment"):
            log_params({"lr": 0.01})
            log_metrics({"accuracy": 0.95})

    Args:
        run_name: Name for the run
        experiment_name: Experiment name (will init if provided)
        tracking_uri: Tracking URI (will init if provided)
        nested: Whether this is a nested run
        tags: Optional tags for the run

    Yields:
        ActiveRun object or None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return

    try:
        if experiment_name:
            init_mlflow(experiment_name, tracking_uri or "mlruns")

        with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
            yield run
    except Exception as e:
        logger.warning(f"MLflow run context failed: {e}")
        yield None


def get_best_run(
    experiment_name: str, metric: str, ascending: bool = False, tracking_uri: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric name to sort by
        ascending: If True, lower is better; if False, higher is better
        tracking_uri: Optional tracking URI

    Returns:
        Dictionary with run info and metrics, or None if not found
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_name}")
            return None

        from mlflow.tracking import MlflowClient

        order = [f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            run_view_type=1,
            max_results=1,
            order_by=order,
        )
        if not runs:
            return None

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.data.tags.get("mlflow.runName"),
            "metrics": dict(best_run.data.metrics),
            "params": dict(best_run.data.params),
            "artifact_uri": best_run.info.artifact_uri,
            "status": best_run.info.status,
            "start_time": best_run.info.start_time,
            "end_time": best_run.info.end_time,
        }
    except Exception as e:
        logger.warning(f"Failed to get best run: {e}")
        return None


def get_run_by_id(run_id: str) -> Optional[dict[str, Any]]:
    """
    Get run details by run ID.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with run info, or None if not found
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        run = client.get_run(run_id)

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
    except Exception as e:
        logger.warning(f"Failed to get run {run_id}: {e}")
        return None


def set_tag(key: str, value: str) -> bool:
    """
    Set a tag on the current run.

    Args:
        key: Tag key
        value: Tag value

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.set_tag(key, value)
        return True
    except Exception as e:
        logger.warning(f"Failed to set tag: {e}")
        return False


def set_tags(tags: dict[str, str]) -> bool:
    """
    Set multiple tags on the current run.

    Args:
        tags: Dictionary of tag key-value pairs

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.set_tags(tags)
        return True
    except Exception as e:
        logger.warning(f"Failed to set tags: {e}")
        return False


def log_text(text: str, artifact_file: str) -> bool:
    """
    Log text content as an artifact.

    Args:
        text: Text content to log
        artifact_file: Filename for the artifact

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.log_text(text, artifact_file)
        return True
    except Exception as e:
        logger.warning(f"Failed to log text: {e}")
        return False


def get_active_run() -> Optional[Any]:
    """
    Get the current active run.

    Returns:
        ActiveRun object or None
    """
    if not MLFLOW_AVAILABLE:
        return None

    return mlflow.active_run()


def get_tracking_uri() -> Optional[str]:
    """
    Get the current tracking URI.

    Returns:
        Tracking URI string or None
    """
    if not MLFLOW_AVAILABLE:
        return None

    return cast(str, mlflow.get_tracking_uri())
