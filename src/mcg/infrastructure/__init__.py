"""Infrastructure wiring for the refactored MCG attack pipeline."""

from __future__ import annotations

import os

import punq

from mcg.config import MetaConditionalAttackParams
from mcg.domain import ADVERSARIAL_BUFFER_KEY, SURROGATE_MODELS_KEY, SURROGATE_OPTIMIZERS_KEY
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
from mcg.domain.services.stats import QueryTracker
from utils.attack_count import AttackCountingFunction
from utils.buffer import AttackListBuffer, ImageBuffer
from utils.surrogate_trainer import TrainModelSurrogate

from mcg.infrastructure.components import (
    AdversarialBufferAdapter,
    CleanBufferAdapter,
    DataLoaderStream,
    FileAttemptLogger,
    GeneratorAdapter,
    ImageBufferAdapter,
    SurrogateTrainerAdapter,
    build_generator,
    create_dataloader,
    initialise_seeds,
    instantiate_attack_strategy,
    load_models_for_dataset,
)

__all__ = ["inject"]


def _resolve_log_path(params: MetaConditionalAttackParams) -> str:
    if params.logging.log_root:
        return params.logging.log_root
    os.makedirs("logs", exist_ok=True)
    targeted_flag = "T" if params.runtime.targeted else "UT"
    filename = f"{params.dataset.name}_{targeted_flag}_{params.models.target}_{params.runtime.method}.log"
    return os.path.join("logs", filename)


def inject(container: punq.Container) -> None:
    params: MetaConditionalAttackParams = container.resolve(MetaConditionalAttackParams)

    initialise_seeds()

    # Data stream
    dataloader = create_dataloader(params)
    container.register(IDataStream, instance=DataLoaderStream(dataloader))

    # Models and optimisers
    target_model, surrogate_models, surrogate_optims = load_models_for_dataset(params)
    container.register(IModel, instance=target_model)
    container.register(SURROGATE_MODELS_KEY, instance=surrogate_models)
    container.register(SURROGATE_OPTIMIZERS_KEY, instance=surrogate_optims)

    # Generator and attack strategy
    generator: GeneratorAdapter = build_generator(params)
    container.register(IGenerator, instance=generator)
    container.register(IAttackStrategy, instance=instantiate_attack_strategy(params))

    # Buffers
    image_buffer = ImageBufferAdapter(ImageBuffer(params.finetune.mini_batch_size))
    clean_buffer = CleanBufferAdapter(ImageBuffer(params.finetune.mini_batch_size))
    container.register(IImageBuffer, instance=image_buffer)
    container.register(ICleanBuffer, instance=clean_buffer)

    adversarial_buffer = None
    if params.finetune.perturbation:
        adv_inner = AttackListBuffer(
            attack_method=params.runtime.method,
            uplimit=params.runtime.buffer_limit,
            batch_size=params.finetune.mini_batch_size,
        )
        adversarial_buffer = AdversarialBufferAdapter(adv_inner)
    container.register(ADVERSARIAL_BUFFER_KEY, instance=adversarial_buffer)

    # Surrogate trainer and query stats
    trainer_adapter = SurrogateTrainerAdapter(TrainModelSurrogate())
    container.register(ISurrogateTrainer, instance=trainer_adapter)

    tracker = QueryTracker(AttackCountingFunction(params.runtime.max_query))
    container.register(IQueryStats, instance=tracker)

    # Logging
    log_path = _resolve_log_path(params)
    container.register(IAttemptLogger, instance=FileAttemptLogger(log_path))
