from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..pt.trainer import CustomTrainer
from ..sft.trainer import CustomSeq2SeqTrainer
from .blockffn_utils import (
    collect_router_tensors,
    compute_activation_locality_loss,
    compute_chunk_sparsity_loss,
    normalize_router_activations,
)

logger = logging.get_logger(__name__)


class _BlockFFNMixIn:
    def _init_blockffn(self) -> None:
        self._blockffn_metrics: dict[str, list[float]] = defaultdict(list)
        self._blockffn_last_log_step: int = -1
        self._blockffn_warned_missing_router: bool = False

    # Utilities -----------------------------------------------------------------

    def _blockffn_enabled(self) -> bool:
        finetuning_args = getattr(self, "finetuning_args", None)
        return bool(finetuning_args and finetuning_args.use_blockffn_loss)

    def _blockffn_router_keys(self) -> Optional[list[str]]:
        finetuning_args = getattr(self, "finetuning_args", None)
        if finetuning_args is None:
            return None
        return finetuning_args.blockffn_router_keys

    def _blockffn_record_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self._blockffn_metrics[key].append(float(value))

    def _blockffn_flush_metrics(self) -> Dict[str, float]:
        averaged: Dict[str, float] = {}
        for key, values in self._blockffn_metrics.items():
            if values:
                averaged[key] = float(sum(values) / len(values))
        self._blockffn_metrics.clear()
        return averaged

    def _blockffn_maybe_log_warning(self) -> None:
        if self._blockffn_warned_missing_router:
            return
        if getattr(self, "state", None) is None or not self.state.is_world_process_zero:
            return
        logger.warning(
            "BlockFFN auxiliary losses are enabled, but no router activations were found in the model outputs. "
            "Ensure that the underlying MoE model exposes router logits/probabilities or enable the appropriate "
            "configuration flag (e.g., `moe_output_router_logits`)."
        )
        self._blockffn_warned_missing_router = True

    # Core computation -----------------------------------------------------------

    def _blockffn_compute_auxiliary(
        self,
        outputs,
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if not self._blockffn_enabled():
            return None, {}

        finetuning_args = self.finetuning_args
        token_mask = labels != IGNORE_INDEX
        router_tensors = collect_router_tensors(outputs, self._blockffn_router_keys())
        router_probs = normalize_router_activations(
            router_tensors,
            temperature=finetuning_args.blockffn_prob_temperature,
            eps=finetuning_args.blockffn_min_prob_eps,
        )

        if not router_probs:
            self._blockffn_maybe_log_warning()
            return None, {}

        locality_loss, locality_metrics = compute_activation_locality_loss(
            router_probs,
            token_mask,
            finetuning_args.blockffn_sigmoid_alpha,
        )
        chunk_loss, chunk_metrics = compute_chunk_sparsity_loss(
            router_probs,
            token_mask,
            finetuning_args.blockffn_chunk_len,
            finetuning_args.blockffn_min_prob_eps,
        )

        total_loss = None
        log_metrics: Dict[str, float] = {}

        if locality_loss is not None and finetuning_args.blockffn_locality_weight > 0.0:
            scaled = finetuning_args.blockffn_locality_weight * locality_loss
            total_loss = scaled if total_loss is None else total_loss + scaled
            log_metrics["blockffn_locality_loss"] = locality_loss.detach().item()

        if chunk_loss is not None and finetuning_args.blockffn_chunk_weight > 0.0:
            scaled = finetuning_args.blockffn_chunk_weight * chunk_loss
            total_loss = scaled if total_loss is None else total_loss + scaled
            log_metrics["blockffn_chunk_loss"] = chunk_loss.detach().item()

        log_metrics.update(locality_metrics)
        log_metrics.update(chunk_metrics)

        return total_loss, log_metrics

    # Logging -------------------------------------------------------------------

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:  # type: ignore[override]
        if self._blockffn_enabled() and self.finetuning_args.blockffn_log_stats and self._blockffn_metrics:
            step = logs.get("step", getattr(self.state, "global_step", 0))
            interval = max(self.finetuning_args.blockffn_log_interval, 1)
            if step % interval == 0 and step != self._blockffn_last_log_step:
                logs.update(self._blockffn_flush_metrics())
                self._blockffn_last_log_step = step
        return super().log(logs, *args, **kwargs)


class BlockFFNSeq2SeqTrainer(_BlockFFNMixIn, CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_blockffn()

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        blockffn_active = self._blockffn_enabled() and self.training and "labels" in inputs
        labels_clone = inputs["labels"].clone() if blockffn_active else None

        need_outputs = return_outputs or blockffn_active
        if need_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if blockffn_active and labels_clone is not None:
            aux_loss, metrics = self._blockffn_compute_auxiliary(outputs, labels_clone)
            if aux_loss is not None:
                loss = loss + aux_loss

            if metrics and self.state.is_world_process_zero:
                self._blockffn_record_metrics(metrics)

        if return_outputs:
            return loss, outputs
        return loss


class BlockFFNTrainer(_BlockFFNMixIn, CustomTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_blockffn()

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        blockffn_active = self._blockffn_enabled() and self.training and "labels" in inputs
        labels_clone = inputs["labels"].clone() if blockffn_active else None

        need_outputs = return_outputs or blockffn_active
        if need_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if blockffn_active and labels_clone is not None:
            aux_loss, metrics = self._blockffn_compute_auxiliary(outputs, labels_clone)
            if aux_loss is not None:
                loss = loss + aux_loss

            if metrics and self.state.is_world_process_zero:
                self._blockffn_record_metrics(metrics)

        if return_outputs:
            return loss, outputs
        return loss

