"""Typed configuration objects for the refactored MCG project.

The dataclasses in this module mirror the original argparse options exposed by
``attack.py`` so that higher-level layers can operate on structured,
validated data rather than untyped ``Namespace`` instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

DatasetName = Literal["imagenet", "cifar10"]
AttackMethodName = Literal["square", "signhunter", "cgattack"]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing the evaluation dataset."""

    name: DatasetName
    root: str
    batch_size: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Target and surrogate model selection."""

    target: str
    surrogates: Sequence[str]
    defence: Optional[str] = None


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters required to instantiate the conditional Glow generator."""

    checkpoint_path: str
    x_size: Tuple[int, int, int] = (3, 224, 224)
    y_size: Tuple[int, int, int] = (3, 224, 224)
    x_hidden_channels: int = 64
    x_hidden_size: int = 128
    y_hidden_channels: int = 256
    flow_depth: int = 8
    num_levels: int = 3
    learn_top: bool = False
    down_sample_x: int = 8
    down_sample_y: int = 8
    y_bins: float = 2.0
    tanh: bool = False


@dataclass(frozen=True)
class AttackRuntimeConfig:
    """Runtime options for executing the black-box attack."""

    method: AttackMethodName
    max_query: int = 10_000
    targeted: bool = False
    target_label: Optional[int] = None
    buffer_limit: int = 1
    test_first_success_only: bool = False


@dataclass(frozen=True)
class FinetuneConfig:
    """Fine-tuning switches influencing how surrogates and generator adapt."""

    clean: bool = False
    perturbation: bool = False
    glow: bool = False
    reload_generator: bool = False
    latent: bool = False
    mini_batch_size: int = 20
    max_grad_clip: float = 5.0


@dataclass(frozen=True)
class LoggingConfig:
    """Logging preferences for attack execution."""

    log_root: Optional[str] = None
    mute_stdout: bool = False


@dataclass(frozen=True)
class MetaConditionalAttackParams:
    """Aggregate configuration bundle consumed by the refactored pipeline."""

    dataset: DatasetConfig
    models: ModelConfig
    generator: GeneratorConfig
    runtime: AttackRuntimeConfig
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    class_num: Optional[int] = None
    linf: Optional[float] = None

    def resolved_class_num(self) -> int:
        """Return the effective number of classes for the dataset."""

        if self.class_num is not None:
            return self.class_num
        if self.dataset.name == "imagenet":
            return 1_000
        if self.dataset.name == "cifar10":
            return 10
        raise ValueError(f"Unsupported dataset: {self.dataset.name}")

    def resolved_linf(self) -> float:
        """Return the perturbation budget associated with the dataset."""

        if self.linf is not None:
            return self.linf
        if self.dataset.name == "imagenet":
            return 0.05
        if self.dataset.name == "cifar10":
            return 8.0 / 255.0
        raise ValueError(f"Unsupported dataset: {self.dataset.name}")


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "GeneratorConfig",
    "AttackRuntimeConfig",
    "FinetuneConfig",
    "LoggingConfig",
    "MetaConditionalAttackParams",
]
