"""Domain-layer protocol definitions for the refactored MCG attack pipeline."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

import torch


class IModel(Protocol):
    """Generic classifier contract expected by the domain orchestrator."""

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def train(self, mode: bool = True): ...

    @abstractmethod
    def eval(self): ...

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    @abstractmethod
    def zero_grad(self): ...


class IGenerator(Protocol):
    """Conditional generator used to produce initial perturbations."""

    def clone_state(self) -> object: ...

    def restore_state(self, state: object) -> None: ...

    def initialize_latent(self, images: torch.Tensor) -> tuple[torch.Tensor, object]: ...

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
    ) -> tuple[torch.Tensor, object]: ...

    def meta_finetune(
        self,
        latents: torch.Tensor,
        latents_aux: object,
        images: torch.Tensor,
        labels: torch.Tensor,
        surrogates: Sequence[IModel],
        *,
        steps: int,
    ) -> None: ...

    def generate(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor: ...


class IAttackStrategy(Protocol):
    """Black-box attack strategy such as Square, SignHunter or CGAttack."""

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        init: torch.Tensor | None,
        latents: torch.Tensor | None,
        loss_fn,
        buffer: "IAdversarialBuffer | None",
    ) -> "AttackResult": ...


class IImageBuffer(Protocol):
    """Stores clean examples and associated logits for later use."""

    def add(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        logits: torch.Tensor,
        score: float,
    ) -> None: ...

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class ICleanBuffer(Protocol):
    """Aggregates clean examples used to fine-tune surrogates."""

    def add(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> bool: ...

    def make_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def clear(self) -> None: ...


class IAdversarialBuffer(Protocol):
    """Maintains historical adversarial examples for lifelong learning."""

    def add_clean(self, images: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None: ...

    def add(self, images: torch.Tensor, logits: torch.Tensor) -> None: ...

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @property
    def capacity(self) -> int: ...

    def __len__(self) -> int: ...


class ISurrogateTrainer(Protocol):
    """Fine-tunes surrogate models based on clean or adversarial samples."""

    def finetune_clean(self, models: Sequence[IModel], optims: Sequence[torch.optim.Optimizer], batch) -> None: ...

    def finetune_adversarial(
        self,
        models: Sequence[IModel],
        optims: Sequence[torch.optim.Optimizer],
        memory_batch,
        current_batch,
    ) -> None: ...


class IQueryStats(Protocol):
    """Tracks query-related statistics throughout the attack."""

    def update(self, queries: int, success: bool) -> None: ...

    def summary(self) -> "QuerySummary": ...


class IAttemptLogger(Protocol):
    """Persists per-attempt metrics to disk and optionally stdout."""

    def log_attempt(self, message: str) -> None: ...

    def flush(self) -> None: ...


class IDataStream(Protocol):
    """Iterable stream producing batches of clean images and labels."""

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]: ...


@dataclass(frozen=True)
class AttackResult:
    success: bool
    query_count: int
    adversarial_image: torch.Tensor
    logits: torch.Tensor
    latent: torch.Tensor | None = None


@dataclass(frozen=True)
class QuerySummary:
    mean_queries: float
    median_queries: float
    first_success_rate: float
    success_rate: float
    total_attempts: int
