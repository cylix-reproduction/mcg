"""Infrastructure adapters bridging legacy MCG modules to the refactored domain."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from attacks.cgattack import CGAttack
from attacks.signhunter import SignHunter
from attacks.square import SquareAttack
from mcg.config import MetaConditionalAttackParams
from mcg.domain.interfaces import (
    IAttackStrategy,
    IAdversarialBuffer,
    IAttemptLogger,
    ICleanBuffer,
    IDataStream,
    IGenerator,
    IImageBuffer,
    IModel,
    ISurrogateTrainer,
    AttackResult,
)
from data import datasets as legacy_datasets
from models.flow_latent import generate_interface, latent_initialize, latent_operate
from utils import attack_init
from utils.buffer import AttackListBuffer, ImageBuffer
from utils.finetune import finetune_latent as legacy_finetune_latent
from utils.finetune import meta_finetune as legacy_meta_finetune
from utils.load_models import load_cifar_model, load_generator, load_imagenet_model
from utils.surrogate_trainer import TrainModelSurrogate


@dataclass
class DataLoaderStream(IDataStream):
    """Stream batches from a ``torch.utils.data.DataLoader``."""

    loader: DataLoader

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for images, labels in self.loader:
            yield images, labels


class ImageBufferAdapter(IImageBuffer):
    """Adapter for ``utils.buffer.ImageBuffer``."""

    def __init__(self, inner: ImageBuffer) -> None:
        self._inner = inner

    def add(self, images: torch.Tensor, labels: torch.Tensor, *, logits: torch.Tensor, score: float) -> None:
        self._inner.add(images, labels, logits=logits, score=score)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inner.sample_batch(batch_size)

    def clear(self) -> None:
        self._inner.clear()

    def __len__(self) -> int:
        return self._inner.length()

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self._inner.get_item(index)
        logits = self._inner.clean_logits[index]
        if not torch.is_tensor(label):
            label = torch.tensor(label)
        return image, label, logits


class CleanBufferAdapter(ICleanBuffer):
    """Adapter exposing ``ImageBuffer`` as a clean buffer."""

    def __init__(self, inner: ImageBuffer) -> None:
        self._inner = inner

    def add(self, images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> bool:
        return bool(self._inner.add(images, labels, logits=logits))

    def make_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inner.make_batch()

    def clear(self) -> None:
        self._inner.clear()


class AdversarialBufferAdapter(IAdversarialBuffer):
    """Adapter for ``AttackListBuffer`` that exposes the expected protocol."""

    def __init__(self, inner: AttackListBuffer) -> None:
        self._inner = inner

    @property
    def raw(self) -> AttackListBuffer:
        return self._inner

    def add_clean(self, images: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._inner.add_clean(images.cpu(), logits.cpu(), labels.cpu())

    def add(self, images: torch.Tensor, logits: torch.Tensor) -> None:
        self._inner.add(images.cpu(), logits.cpu())

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inner.sample_batch(batch_size)

    @property
    def capacity(self) -> int:
        return self._inner.buffer_limit

    def __len__(self) -> int:
        return self._inner.length()


class SurrogateTrainerAdapter(ISurrogateTrainer):
    """Bridge ``TrainModelSurrogate`` to the domain protocol."""

    def __init__(self, trainer: TrainModelSurrogate) -> None:
        self._trainer = trainer

    def finetune_clean(
        self,
        models: Sequence[IModel],
        optims: Sequence[torch.optim.Optimizer],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        images, logits, labels = batch
        for model, optim in zip(models, optims):
            self._trainer.forward_loss(model, optim, images, logits, labels)

    def finetune_adversarial(
        self,
        models: Sequence[IModel],
        optims: Sequence[torch.optim.Optimizer],
        memory_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        current_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        for model, optim in zip(models, optims):
            self._trainer.lifelong_forward_loss(model, optim, memory_batch, current_batch)


class FileAttemptLogger(IAttemptLogger):
    """Write attempt logs to disk and optionally stdout."""

    def __init__(self, path: str) -> None:
        self._path = path

    def log_attempt(self, message: str) -> None:
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(message)

    def flush(self) -> None:  # pragma: no cover - nothing buffered
        return None


class GeneratorAdapter(IGenerator):
    """Wrap the legacy conditional Glow generator with helper utilities."""

    def __init__(self, generator, params: MetaConditionalAttackParams) -> None:
        self._generator = generator
        self._params = params
        self._args = SimpleNamespace(
            linf=params.resolved_linf(),
            targeted=params.runtime.targeted,
            class_num=params.resolved_class_num(),
            max_grad_clip=params.finetune.max_grad_clip,
        )
        self._generate_fn = generate_interface(generator, latent_operate, self._args.linf)

    def clone_state(self) -> dict:
        return copy.deepcopy(self._generator.state_dict())

    def restore_state(self, state: dict) -> None:
        self._generator.load_state_dict(copy.deepcopy(state))
        self._generator.eval()

    def initialize_latent(self, images: torch.Tensor) -> tuple[torch.Tensor, object]:
        return latent_initialize(images, self._generator, latent_operate)

    def finetune_latent(
        self,
        latents: torch.Tensor,
        latents_aux: object,
        images: torch.Tensor,
        labels: torch.Tensor,
        surrogates: Sequence[IModel],
        *,
        iterations: int = 10,
        lr: float = 0.01,
    ) -> tuple[torch.Tensor, object]:
        return legacy_finetune_latent(
            self._generator,
            surrogates,
            images,
            labels,
            latents,
            self._args,
            iteration=iterations,
            lr=lr,
        )

    def meta_finetune(
        self,
        latents: torch.Tensor,
        latents_aux: object,
        images: torch.Tensor,
        labels: torch.Tensor,
        surrogates: Sequence[IModel],
        *,
        steps: int,
    ) -> None:
        legacy_meta_finetune(
            self._generator,
            surrogates,
            images,
            labels,
            latents,
            self._args,
            meta_iteration=steps,
        )

    def generate(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self._generate_fn(images, latents)


class AttackStrategyAdapter(IAttackStrategy):
    """Wrap legacy attack implementations and expose a unified interface."""

    def __init__(self, method: str, attack_impl) -> None:
        self._method = method
        self._attack_impl = attack_impl

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        init: torch.Tensor | None,
        latents: torch.Tensor | None,
        loss_fn,
        buffer: IAdversarialBuffer | None,
    ) -> AttackResult:
        raw_buffer = None
        if buffer is not None:
            raw_buffer = buffer.raw if isinstance(buffer, AdversarialBufferAdapter) else buffer

        kwargs: dict = {}
        if self._method == "cgattack":
            kwargs["latent"] = latents
        else:
            kwargs["init"] = init

        result = self._attack_impl.attack(
            loss_fn,
            images,
            labels,
            buffer=raw_buffer,
            **kwargs,
        )
        logits = result.get("logits_best")
        if logits is None and "logits" in result:
            logits = result["logits"]
        return AttackResult(
            success=bool(result.get("success", False)),
            query_count=int(result.get("query_cnt", 0)),
            adversarial_image=result.get("adv", images),
            logits=logits,
            latent=result.get("latent"),
        )


def load_models_for_dataset(params: MetaConditionalAttackParams) -> tuple[IModel, list[IModel], list[optim.Optimizer]]:
    if params.dataset.name == "imagenet":
        loader = load_imagenet_model
    elif params.dataset.name == "cifar10":
        loader = load_cifar_model
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"Unsupported dataset {params.dataset.name}")

    target = loader(params.models.target, defence_method=params.models.defence)
    surrogates: list[IModel] = []
    optimizers: list[optim.Optimizer] = []
    for surrogate_name in params.models.surrogates:
        model, optim = loader(surrogate_name, require_optim=True)
        surrogates.append(model)
        optimizers.append(optim)
    return target, surrogates, optimizers


def instantiate_attack_strategy(params: MetaConditionalAttackParams) -> AttackStrategyAdapter:
    dataset = params.dataset.name
    linf = params.resolved_linf()
    class_num = params.resolved_class_num()
    targeted = params.runtime.targeted
    max_query = params.runtime.max_query

    if params.runtime.method == "square":
        attack = SquareAttack(dataset, max_query, targeted, class_num, linf=linf)
    elif params.runtime.method == "signhunter":
        attack = SignHunter(dataset, max_query, targeted, class_num, linf=linf)
    elif params.runtime.method == "cgattack":
        attack = CGAttack(dataset, max_query, targeted, class_num, linf=linf)
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"Unknown attack method {params.runtime.method}")
    return AttackStrategyAdapter(params.runtime.method, attack)


def build_generator(params: MetaConditionalAttackParams):
    generator = load_generator(
        SimpleNamespace(
            generator_path=params.generator.checkpoint_path,
            x_size=params.generator.x_size,
            y_size=params.generator.y_size,
            x_hidden_channels=params.generator.x_hidden_channels,
            x_hidden_size=params.generator.x_hidden_size,
            y_hidden_channels=params.generator.y_hidden_channels,
            flow_depth=params.generator.flow_depth,
            num_levels=params.generator.num_levels,
            learn_top=params.generator.learn_top,
            y_bins=params.generator.y_bins,
            down_sample_x=params.generator.down_sample_x,
            down_sample_y=params.generator.down_sample_y,
            tanh=params.generator.tanh,
        )
    )
    return GeneratorAdapter(generator, params)


def create_dataloader(params: MetaConditionalAttackParams) -> DataLoader:
    if params.dataset.name == "imagenet":
        dataset = legacy_datasets.imagenet(params.dataset.root, mode="validation")
    elif params.dataset.name == "cifar10":
        dataset = legacy_datasets.cifar10(params.dataset.root, mode="validation")
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"Unsupported dataset {params.dataset.name}")
    return DataLoader(dataset, batch_size=params.dataset.batch_size, shuffle=False, drop_last=False)


def initialise_seeds() -> None:
    attack_init.seed_init()
