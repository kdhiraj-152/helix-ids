"""freeze_manager: Backbone freeze/unfreeze state management.

Phase 13A-2 extraction from HelixFullTrainer.

FreezeManager provides:
    - Backbone freeze state tracking
    - Freeze/unfreeze decision logic
    - Backbone parameter list management
"""


class FreezeManager:
    """Manages backbone freeze state and freeze/unfreeze decisions.

    The manager tracks freeze state and provides decision helpers.
    Trainer wrappers handle actual model parameter mutation and logging.
    """

    def __init__(self) -> None:
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    @backbone_frozen.setter
    def backbone_frozen(self, value: bool) -> None:
        self._backbone_frozen = bool(value)

    def should_unfreeze(self, global_step: int, unfreeze_backbone_step: int) -> bool:
        """Return True if backbone should be unfrozen based on step schedule."""
        if not self._backbone_frozen:
            return False
        if unfreeze_backbone_step <= 0:
            return False
        return int(global_step) >= int(unfreeze_backbone_step)
