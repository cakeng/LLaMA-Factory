from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

DEFAULT_ROUTER_KEYS: Sequence[str] = (
    "router_logits",
    "router_probs",
    "router_probabilities",
    "gate_logits",
    "gating_logits",
)


def _flatten_tensors(value) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_flatten_tensors(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(_flatten_tensors(item))

    return tensors


def collect_router_tensors(outputs, router_keys: Iterable[str] | None = None) -> list[torch.Tensor]:
    keys = list(router_keys) if router_keys is not None else list(DEFAULT_ROUTER_KEYS)
    tensors: list[torch.Tensor] = []

    if isinstance(outputs, dict):
        lookup = outputs
    else:
        lookup = {k: getattr(outputs, k) for k in dir(outputs) if not k.startswith("_")}

    for key in keys:
        if key not in lookup:
            continue

        tensors.extend(_flatten_tensors(lookup[key]))

    return tensors


def normalize_router_activations(
    tensors: Sequence[torch.Tensor],
    temperature: float,
    eps: float,
) -> list[torch.Tensor]:
    probs: list[torch.Tensor] = []
    if not tensors:
        return probs

    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue

        data = tensor.to(torch.float32)
        if data.dim() < 3:
            continue

        if temperature != 1.0:
            data = data / temperature

        # Softmax produces probabilities even if the router already returned them.
        norm = torch.softmax(data, dim=-1)
        probs.append(torch.clamp(norm, min=eps))

    return probs


def _estimate_token_sparsity(probs: torch.Tensor, mask: torch.Tensor, threshold: float = 0.01) -> torch.Tensor | None:
    masked = mask.unsqueeze(-1).float()
    denom = masked.sum()
    if denom == 0:
        return None

    active = (probs > threshold).float()
    ratio = (active * masked).sum() / (denom * probs.size(-1))
    return 1.0 - ratio


def compute_activation_locality_loss(
    probs_list: Sequence[torch.Tensor],
    token_mask: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not probs_list:
        return None, {}

    losses: list[torch.Tensor] = []
    tls_values: list[torch.Tensor] = []

    for probs in probs_list:
        if probs.size(1) < 2:
            continue

        mask_forward = token_mask[:, :-1] & token_mask[:, 1:]
        if not mask_forward.any():
            continue

        sharpen_curr = torch.sigmoid(alpha * (probs[:, :-1, :] - 0.5))
        sharpen_next = torch.sigmoid(alpha * (probs[:, 1:, :] - 0.5))
        bce = F.binary_cross_entropy(sharpen_next, sharpen_curr, reduction="none").mean(dim=-1)

        loss = (bce * mask_forward.float()).sum() / mask_forward.float().sum()
        losses.append(loss)

        tls = _estimate_token_sparsity(probs, token_mask)
        if tls is not None:
            tls_values.append(tls.detach())

    if not losses:
        return None, {}

    metrics: dict[str, float] = {}
    if tls_values:
        metrics["blockffn_tls"] = torch.stack(tls_values).mean().item()

    return torch.stack(losses).mean(), metrics


def compute_chunk_sparsity_loss(
    probs_list: Sequence[torch.Tensor],
    token_mask: torch.Tensor,
    chunk_len: int,
    eps: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if chunk_len <= 0 or not probs_list:
        return None, {}

    losses: list[torch.Tensor] = []
    cls_values: list[torch.Tensor] = []

    for probs in probs_list:
        if probs.size(1) < chunk_len:
            continue

        unfolded_probs = probs.unfold(dimension=1, size=chunk_len, step=1)
        unfolded_mask = token_mask.float().unfold(dimension=1, size=chunk_len, step=1)

        valid_mask = (unfolded_mask.sum(-1) == float(chunk_len))
        if not valid_mask.any():
            continue

        # union probability of expert activation inside each chunk
        complement = torch.clamp(1.0 - unfolded_probs, min=eps, max=1.0)
        union = 1.0 - torch.prod(complement, dim=-2)
        union_mean = union.mean(dim=-1)

        loss = (union_mean * valid_mask.float()).sum() / valid_mask.float().sum()
        losses.append(loss)

        denom = valid_mask.float().sum()
        if denom > 0:
            sparsity = 1.0 - (union * valid_mask.unsqueeze(-1).float()).sum() / (denom * union.size(-1))
            cls_values.append(sparsity.detach())

    if not losses:
        return None, {}

    metrics: dict[str, float] = {}
    if cls_values:
        metrics["blockffn_cls"] = torch.stack(cls_values).mean().item()

    return torch.stack(losses).mean(), metrics

