"""Domain adaptation modules for handling train-test distribution shift."""

from .combined_da import (
    CombinedDAConfig,
    CombinedDANNLoss,
    CombinedDomainAdaptation,
    create_combined_da_loss,
)
from .coral_loss import CombinedAlignmentLoss, CORALLoss, compute_coral
from .dann import DANN, DANNConfig, DANNLoss, GradientReversalLayer
from .label_aware_da import (
    ClassConditionalMMDLoss,
    ConditionalDomainDiscriminator,
    LabelAwareDAConfig,
    LabelAwareDALoss,
    LabelAwareDANN,
    PartialTransferReweighter,
    create_label_aware_dann,
)
from .mmd_loss import MMDLoss, compute_mmd
from .transfer_learning import (
    FeatureAligner,
    MultiDatasetPretrainer,
    TransferLearningConfig,
    create_pretrainer,
)

__all__ = [
    "DANN",
    "GradientReversalLayer",
    "DANNConfig",
    "DANNLoss",
    "MMDLoss",
    "compute_mmd",
    "CORALLoss",
    "compute_coral",
    "CombinedAlignmentLoss",
    "CombinedDomainAdaptation",
    "CombinedDAConfig",
    "CombinedDANNLoss",
    "create_combined_da_loss",
    "LabelAwareDANN",
    "LabelAwareDAConfig",
    "LabelAwareDALoss",
    "ConditionalDomainDiscriminator",
    "PartialTransferReweighter",
    "ClassConditionalMMDLoss",
    "create_label_aware_dann",
    "MultiDatasetPretrainer",
    "TransferLearningConfig",
    "FeatureAligner",
    "create_pretrainer",
]
