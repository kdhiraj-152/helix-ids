"""Platform-specific configuration loader for HELIX-IDS."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PlatformConfig:
    """Configuration for a specific deployment platform."""

    name: str
    max_size_kb: int
    max_latency_ms: float
    max_params: int
    model_type: str
    hidden_dims: Optional[tuple[int, ...]]
    dropout: float
    binary_only: bool
    family_head: bool
    finegrain_head: bool
    target_accuracy: float


def load_platform_config(platform: str) -> PlatformConfig:
    """Load configuration for a specific platform.

    Args:
        platform: Platform name (esp32, rpi_zero, rpi_4, server)

    Returns:
        PlatformConfig with settings for the specified platform

    Raises:
        KeyError: If platform is not found in configuration
        FileNotFoundError: If configuration file is missing
    """
    config_path = Path(__file__).parents[3] / "config" / "platform_configs.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    p = configs["platforms"][platform]
    return PlatformConfig(
        name=platform,
        max_size_kb=p["constraints"].get(
            "max_size_kb", p["constraints"].get("max_size_mb", 1) * 1024
        ),
        max_latency_ms=p["constraints"].get(
            "max_latency_ms", p["constraints"].get("max_latency_us", 1000) / 1000
        ),
        max_params=p["constraints"]["max_params"],
        model_type=p["model"]["type"],
        hidden_dims=tuple(p["model"].get("hidden_dims") or []) or None,
        dropout=p["model"].get("dropout", 0.0),
        binary_only=p["classification"]["binary_only"],
        family_head=p["classification"]["family_head"],
        finegrain_head=p["classification"]["finegrain_head"],
        target_accuracy=p["targets"]["binary_accuracy"],
    )
