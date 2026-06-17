"""pytest conftest — register Hypothesis 'dev' (fast), 'ci', and 'thorough' profiles."""
import hypothesis


def pytest_configure(config):
    """Register Hypothesis profiles for different environments.

    Use: pytest --hypothesis-profile=ci
         pytest --hypothesis-profile=thorough (1000 examples)
         pytest --hypothesis-profile=dev (default, 50 examples)
    """
    hypothesis.settings.register_profile(
        "ci",
        max_examples=50,
        deadline=1000,
        suppress_health_check=list(hypothesis.HealthCheck),
    )
    hypothesis.settings.register_profile(
        "thorough",
        max_examples=1000,
        deadline=None,
        suppress_health_check=list(hypothesis.HealthCheck),
    )
    hypothesis.settings.register_profile(
        "dev",
        max_examples=50,
        deadline=None,
        suppress_health_check=list(hypothesis.HealthCheck),
    )
