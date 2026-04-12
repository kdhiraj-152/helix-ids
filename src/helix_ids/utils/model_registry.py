"""
MLflow Model Registry Utilities for HELIX-IDS

Provides model registry functions for managing model versions and promotions.

Functions:
    - register_model: Register a model to the MLflow Model Registry
    - promote_model: Promote a model version to a new stage
    - get_latest_production_model: Get the latest production model
    - compare_models: Compare metrics across multiple runs
    - transition_model_version: Transition model version stage
"""

import logging
import importlib
from typing import Any, Optional, cast

# Check if MLflow is available
try:
    mlflow = importlib.import_module("mlflow")
    MlflowClient = importlib.import_module("mlflow.tracking").MlflowClient
    ModelVersionStatus = importlib.import_module(
        "mlflow.entities.model_registry.model_version_status"
    ).ModelVersionStatus

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = cast(Any, None)
    MlflowClient = cast(Any, None)
    ModelVersionStatus = cast(Any, None)

logger = logging.getLogger(__name__)
MLFLOW_UNAVAILABLE_MSG = "MLflow not available."


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


def register_model(
    model_uri: str,
    name: str,
    tags: Optional[dict[str, str]] = None,
    description: Optional[str] = None,
    await_registration: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Register a model to the MLflow Model Registry.

    Args:
        model_uri: URI of the model to register (e.g., "runs:/<run_id>/model")
        name: Name to register the model under
        tags: Optional tags for the model version
        description: Optional description for the model version
        await_registration: Wait for registration to complete

    Returns:
        Dictionary with model version info, or None on failure
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(f"{MLFLOW_UNAVAILABLE_MSG} Cannot register model.")
        return None

    try:
        client = MlflowClient()

        # Register the model
        result = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            tags=tags,
            await_registration_for=300 if await_registration else 0,
        )

        # Update description if provided
        if description:
            client.update_model_version(name=name, version=str(result.version), description=description)

        logger.info(f"Registered model: {name} version {result.version}")

        return {
            "name": result.name,
            "version": result.version,
            "creation_timestamp": result.creation_timestamp,
            "status": result.status,
            "source": result.source,
            "run_id": result.run_id,
        }
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def promote_model(
    name: str,
    version: int,
    stage: str,
    archive_existing: bool = True,
    description: Optional[str] = None,
) -> bool:
    """
    Promote a model version to a new stage.

    Args:
        name: Registered model name
        version: Model version number
        stage: Target stage ("Staging", "Production", "Archived", "None")
        archive_existing: Whether to archive existing models in target stage
        description: Optional description update

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(f"{MLFLOW_UNAVAILABLE_MSG} Cannot promote model.")
        return False

    valid_stages = ["Staging", "Production", "Archived", "None"]
    if stage not in valid_stages:
        logger.error(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        return False

    try:
        client = MlflowClient()

        # Transition the model version
        client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=archive_existing
        )

        # Update description if provided
        if description:
            client.update_model_version(name=name, version=str(version), description=description)

        logger.info(f"Promoted {name} v{version} to {stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def get_latest_production_model(name: str, stage: str = "Production") -> Optional[dict[str, Any]]:
    """
    Get the latest model version in a specific stage.

    Args:
        name: Registered model name
        stage: Stage to query ("Production", "Staging", etc.)

    Returns:
        Dictionary with model version info, or None if not found
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(f"{MLFLOW_UNAVAILABLE_MSG} Cannot get model.")
        return None

    try:
        client = MlflowClient()

        # Get latest versions for the stage
        versions = client.get_latest_versions(name, stages=[stage])

        if not versions:
            logger.info(f"No {stage} versions found for model: {name}")
            return None

        # Get the latest version
        latest = versions[0]

        return {
            "name": latest.name,
            "version": latest.version,
            "stage": latest.current_stage,
            "description": latest.description,
            "source": latest.source,
            "run_id": latest.run_id,
            "status": latest.status,
            "creation_timestamp": latest.creation_timestamp,
            "last_updated_timestamp": latest.last_updated_timestamp,
        }
    except Exception as e:
        logger.error(f"Failed to get production model: {e}")
        return None


def compare_models(run_ids: list[str], metrics: list[str]) -> Optional[dict[str, Any]]:  # NOSONAR
    """
    Compare metrics across multiple runs.

    Args:
        run_ids: List of MLflow run IDs to compare
        metrics: List of metric names to compare

    Returns:
        Dictionary with comparison results, or None on failure
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(f"{MLFLOW_UNAVAILABLE_MSG} Cannot compare models.")
        return None

    if not run_ids:
        logger.warning("No run IDs provided for comparison")
        return None

    try:
        client = MlflowClient()

        comparison: dict[str, Any] = {
            "runs": [],
            "best_per_metric": {},
        }

        # Collect metrics for each run
        for run_id in run_ids:
            run = client.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "metrics": {},
                "params": dict(run.data.params),
                "status": run.info.status,
            }

            for metric in metrics:
                value = run.data.metrics.get(metric)
                if value is not None:
                    run_data["metrics"][metric] = value

            comparison["runs"].append(run_data)

        # Find best run for each metric (assuming higher is better)
        for metric in metrics:
            best_value = None
            best_run_id = None

            for run_data in comparison["runs"]:
                value = run_data["metrics"].get(metric)
                if value is not None:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_run_id = run_data["run_id"]

            if best_run_id:
                comparison["best_per_metric"][metric] = {
                    "run_id": best_run_id,
                    "value": best_value,
                }

        return comparison
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return None


def list_model_versions(
    name: str, stages: Optional[list[str]] = None
) -> Optional[list[dict[str, Any]]]:
    """
    List all versions of a registered model.

    Args:
        name: Registered model name
        stages: Optional list of stages to filter by

    Returns:
        List of model version dictionaries, or None on failure
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return None

    try:
        client = MlflowClient()

        if stages:
            versions = client.get_latest_versions(name, stages=stages)
        else:
            versions = client.search_model_versions(f"name='{name}'")

        return [
            {
                "name": v.name,
                "version": v.version,
                "stage": v.current_stage,
                "description": v.description,
                "source": v.source,
                "run_id": v.run_id,
                "status": v.status,
                "creation_timestamp": v.creation_timestamp,
            }
            for v in versions
        ]
    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        return None


def delete_model_version(name: str, version: int) -> bool:
    """
    Delete a specific model version.

    Args:
        name: Registered model name
        version: Version number to delete

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return False

    try:
        client = MlflowClient()
        client.delete_model_version(name=name, version=str(version))
        logger.info(f"Deleted {name} version {version}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model version: {e}")
        return False


def get_model_version_by_alias(name: str, alias: str) -> Optional[dict[str, Any]]:
    """
    Get model version by alias.

    Args:
        name: Registered model name
        alias: Model alias (e.g., "champion", "challenger")

    Returns:
        Model version info or None
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return None

    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(name, alias)

        return {
            "name": version.name,
            "version": version.version,
            "aliases": version.aliases,
            "stage": version.current_stage,
            "source": version.source,
            "run_id": version.run_id,
        }
    except Exception as e:
        logger.error(f"Failed to get model by alias: {e}")
        return None


def set_model_version_alias(name: str, version: int, alias: str) -> bool:
    """
    Set an alias for a model version.

    Args:
        name: Registered model name
        version: Model version number
        alias: Alias to set

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return False

    try:
        client = MlflowClient()
        client.set_registered_model_alias(name, alias, str(version))
        logger.info(f"Set alias '{alias}' for {name} v{version}")
        return True
    except Exception as e:
        logger.error(f"Failed to set alias: {e}")
        return False


def load_model_from_registry(
    name: str,
    version: Optional[int] = None,
    stage: Optional[str] = None,
    alias: Optional[str] = None,
) -> Optional[Any]:
    """
    Load a model from the registry.

    Args:
        name: Registered model name
        version: Specific version to load
        stage: Stage to load from (e.g., "Production")
        alias: Model alias to load

    Returns:
        Loaded model or None
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return None

    try:
        if alias:
            model_uri = f"models:/{name}@{alias}"
        elif version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            model_uri = f"models:/{name}/latest"

        # Try PyTorch flavor
        try:
            mlflow_pytorch = importlib.import_module("mlflow.pytorch")
            model = mlflow_pytorch.load_model(model_uri)
        except Exception:
            # Fallback to generic pyfunc
            model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Loaded model from: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def create_registered_model(
    name: str, tags: Optional[dict[str, str]] = None, description: Optional[str] = None
) -> bool:
    """
    Create a new registered model (without versions).

    Args:
        name: Name for the registered model
        tags: Optional tags
        description: Optional description

    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(MLFLOW_UNAVAILABLE_MSG)
        return False

    try:
        client = MlflowClient()
        client.create_registered_model(name=name, tags=tags, description=description)
        logger.info(f"Created registered model: {name}")
        return True
    except Exception as e:
        # Model might already exist
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            logger.info(f"Registered model already exists: {name}")
            return True
        logger.error(f"Failed to create registered model: {e}")
        return False


def get_model_download_uri(name: str, version: int) -> Optional[str]:
    """
    Get the download URI for a model version.

    Args:
        name: Registered model name
        version: Model version number

    Returns:
        Download URI or None
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available.")
        return None

    try:
        client = MlflowClient()
        version_info = client.get_model_version(name, str(version))
        return str(version_info.source)
    except Exception as e:
        logger.error(f"Failed to get model URI: {e}")
        return None
