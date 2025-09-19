"""Domain orchestrator implementing the refactored MCG attack pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
from loguru import logger

from mcg.config import MetaConditionalAttackParams
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


@dataclass(frozen=True)
class AttackOutcome:
    success: bool
    queries: int


class MetaConditionalAttack:
    """Coordinates dataset iteration, fine-tuning and query-based attack execution."""

    def __init__(
        self,
        *,
        params: MetaConditionalAttackParams,
        data_stream: IDataStream,
        target_model: IModel,
        surrogate_models: Sequence[IModel],
        surrogate_optimizers: Sequence[torch.optim.Optimizer],
        generator: IGenerator,
        attack_strategy: IAttackStrategy,
        image_buffer: IImageBuffer,
        clean_buffer: ICleanBuffer,
        adversarial_buffer,
        trainer: ISurrogateTrainer,
        query_stats: IQueryStats,
        attempt_logger: IAttemptLogger,
    ) -> None:
        self._params = params
        self._data_stream = data_stream
        self._target_model = target_model
        self._surrogates = list(surrogate_models)
        self._surrogate_optimizers = list(surrogate_optimizers)
        self._generator = generator
        self._attack_strategy = attack_strategy
        self._image_buffer = image_buffer
        self._clean_buffer = clean_buffer
        self._adversarial_buffer = adversarial_buffer
        self._trainer = trainer
        self._query_stats = query_stats
        self._attempt_logger = attempt_logger
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self) -> AttackOutcome:
        """Execute the refactored attack pipeline and report aggregate stats."""
        logger.info("Starting MCG attack pipeline on {}", self._params.dataset.name)
        self._collect_clean_buffer()
        self._clean_buffer.clear()

        self._attack_buffered_images()

        summary = self._query_stats.summary()
        logger.info(
            "Attack finished. mean/median queries: {:.2f}/{:.2f}, success rate: {:.2f}%",
            summary.mean_queries,
            summary.median_queries,
            100.0 * summary.success_rate,
        )
        self._attempt_logger.flush()
        return AttackOutcome(success=summary.success_rate > 0.0, queries=int(summary.mean_queries))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_clean_buffer(self) -> None:
        """Populate buffers with correctly classified clean images."""
        self._target_model.eval()
        for images, labels in self._data_stream:
            images = images.to(self._device)
            labels = labels.to(self._device)
            with torch.no_grad():
                logits = torch.softmax(self._target_model(images), dim=1)
            predictions = torch.argmax(logits, dim=1)
            correct_mask = predictions == labels
            if not bool(correct_mask.all()):
                continue

            score = float(logits[0, labels.item()].item()) if logits.ndim == 2 else 0.0
            self._image_buffer.add(images, labels, logits=logits, score=score)

            if self._params.finetune.clean:
                filled = self._clean_buffer.add(images, labels, logits)
                if filled:
                    batch = self._clean_buffer.make_batch()
                    self._trainer.finetune_clean(self._surrogates, self._surrogate_optimizers, batch)
                    self._clean_buffer.clear()

    def _attack_buffered_images(self) -> None:
        if len(self._image_buffer) == 0:
            logger.warning("No clean images satisfied the target model.")
            return

        clone_state = self._generator.clone_state()
        loss_fn = self._create_loss_function()
        for index in range(len(self._image_buffer)):
            torch.cuda.empty_cache()
            image, label, logits = self._image_buffer.get(index)
            clean_image = image.unsqueeze(0).to(self._device)
            clean_logits = logits.to(self._device)
            label_tensor = label.to(self._device)
            if label_tensor.ndim == 0:
                label_tensor = label_tensor.view(1)
            label_tensor = label_tensor.long()

            outcome = self._run_single_attack(
                clean_image,
                label_tensor,
                clean_logits,
                loss_fn,
                clone_state,
                image_index=index,
            )
            self._query_stats.update(outcome.queries, outcome.success)
            self._log_attempt(image_index=index, queries=outcome.queries, success=outcome.success)

        # ensure generator is restored to its original weights
        self._generator.restore_state(clone_state)

    def _run_single_attack(
        self,
        clean_image: torch.Tensor,
        label_tensor: torch.Tensor,
        clean_logits: torch.Tensor,
        loss_fn,
        baseline_state: object,
        *,
        image_index: int,
    ) -> AttackOutcome:
        params = self._params
        finetune = params.finetune
        runtime = params.runtime

        labels_for_attack = label_tensor.clone()
        if runtime.targeted:
            if params.runtime.target_label is None:
                raise ValueError("Targeted attack requires 'target_label' to be set in configuration.")
            if int(label_tensor.item()) == int(runtime.target_label):
                logger.debug("Skipping sample {} because clean label equals target label.", image_index)
                return AttackOutcome(success=False, queries=0)
            labels_for_attack = torch.tensor([runtime.target_label], device=label_tensor.device, dtype=torch.long)
            label_tensor = labels_for_attack

        if finetune.perturbation and self._adversarial_buffer is not None:
            mini_batch = max(1, finetune.mini_batch_size)
            if len(self._image_buffer) >= mini_batch - 1 and mini_batch > 1:
                sampled_images, sampled_logits, sampled_labels = self._image_buffer.sample(mini_batch - 1)
                sampled_images = sampled_images.to(clean_image.device)
                sampled_logits = sampled_logits.to(clean_image.device)
                sampled_labels = sampled_labels.to(clean_image.device).long()
                batch_images = torch.cat([sampled_images, clean_image], dim=0)
                batch_logits = torch.cat([sampled_logits, clean_logits.unsqueeze(0)], dim=0)
                batch_labels = torch.cat([sampled_labels, label_tensor], dim=0)
                self._trainer.finetune_clean(
                    self._surrogates,
                    self._surrogate_optimizers,
                    (batch_images, batch_logits, batch_labels),
                )

            if len(self._adversarial_buffer) > mini_batch:
                mem_images, mem_logits, mem_labels = self._adversarial_buffer.sample(mini_batch)
                mem_images = mem_images.to(clean_image.device)
                mem_logits = mem_logits.to(clean_image.device)
                mem_labels = mem_labels.to(clean_image.device).long()
                memory_batch = (mem_images, mem_logits, mem_labels)
                current_batch = (clean_image, clean_logits.unsqueeze(0), label_tensor)
                self._trainer.finetune_adversarial(
                    self._surrogates,
                    self._surrogate_optimizers,
                    memory_batch,
                    current_batch,
                )

            self._adversarial_buffer.add_clean(clean_image, clean_logits, label_tensor)

        latent, latent_aux = self._generator.initialize_latent(clean_image)
        if finetune.latent:
            latent, latent_aux = self._generator.finetune_latent(
                latent,
                latent_aux,
                clean_image,
                labels_for_attack,
                self._surrogates,
                iterations=10,
                lr=0.01,
            )

        if finetune.glow:
            self._generator.meta_finetune(
                latent,
                latent_aux,
                clean_image,
                labels_for_attack,
                self._surrogates,
                steps=2,
            )

        perturbation = self._generator.generate(clean_image, latent)
        adversarial = torch.clamp(clean_image + perturbation.view_as(clean_image), 0.0, 1.0)
        loss_output = loss_fn(adversarial, labels_for_attack, targeted=runtime.targeted)
        success = bool(loss_output["margin"].le(0).item()) if "margin" in loss_output else False
        query_cnt = 1

        if not runtime.test_first_success_only and not success:
            attack_result = self._attack_strategy.run(
                clean_image,
                labels_for_attack,
                init=perturbation if runtime.method != "cgattack" else None,
                latents=latent if runtime.method == "cgattack" else None,
                loss_fn=loss_fn,
                buffer=self._adversarial_buffer,
            )
            query_cnt += attack_result.query_count
            success = attack_result.success
            adversarial = attack_result.adversarial_image
            if finetune.perturbation and self._adversarial_buffer is not None and attack_result.logits is not None:
                self._adversarial_buffer.add(attack_result.adversarial_image, attack_result.logits)

        if finetune.reload_generator:
            self._generator.restore_state(copy.deepcopy(baseline_state))

        return AttackOutcome(success=success, queries=query_cnt)

    def _create_loss_function(self):
        from attacks.base_attack import margin_loss_interface

        return margin_loss_interface(self._target_model, class_num=self._params.resolved_class_num())

    def _log_attempt(self, *, image_index: int, queries: int, success: bool) -> None:
        stats = self._query_stats.summary()
        message = (
            f"image: {image_index} query_cnt: {queries} success: {success} "
            f"Mean: {stats.mean_queries:.2f} Median: {stats.median_queries:.2f} "
            f"FASR: {stats.first_success_rate:.3f} ASR: {stats.success_rate:.3f}"
        )
        if not self._params.logging.mute_stdout:
            logger.info(message)
        self._attempt_logger.log_attempt(message + "\n")
