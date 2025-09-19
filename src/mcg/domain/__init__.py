"""Domain layer wiring for the refactored MCG attack pipeline."""

from __future__ import annotations

import punq

from mcg.config import MetaConditionalAttackParams
from mcg.domain.attack import MetaConditionalAttack
from mcg.domain.interfaces import (
    IAttackStrategy,
    IAttemptLogger,
    ICleanBuffer,
    IDataStream,
    IGenerator,
    IImageBuffer,
    IModel,
    IQueryStats,
    ISurrogateTrainer,
)

__all__ = ["inject"]


SURROGATE_MODELS_KEY = "surrogate_models"
SURROGATE_OPTIMIZERS_KEY = "surrogate_optimizers"
ADVERSARIAL_BUFFER_KEY = "adversarial_buffer"


def inject(container: punq.Container) -> None:
    """Register domain services in the dependency injection container."""
    params: MetaConditionalAttackParams = container.resolve(MetaConditionalAttackParams)

    orchestrator = MetaConditionalAttack(
        params=params,
        data_stream=container.resolve(IDataStream),
        target_model=container.resolve(IModel),
        surrogate_models=container.resolve(SURROGATE_MODELS_KEY),
        surrogate_optimizers=container.resolve(SURROGATE_OPTIMIZERS_KEY),
        generator=container.resolve(IGenerator),
        attack_strategy=container.resolve(IAttackStrategy),
        image_buffer=container.resolve(IImageBuffer),
        clean_buffer=container.resolve(ICleanBuffer),
        adversarial_buffer=container.resolve(ADVERSARIAL_BUFFER_KEY),
        trainer=container.resolve(ISurrogateTrainer),
        query_stats=container.resolve(IQueryStats),
        attempt_logger=container.resolve(IAttemptLogger),
    )

    container.register(MetaConditionalAttack, instance=orchestrator)
*** End Patch
