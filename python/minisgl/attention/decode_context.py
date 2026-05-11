from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

DecodeContextMode = Literal["dense", "recent", "uniform", "random", "prefix_recent"]


@dataclass(frozen=True)
class DecodeContextConfig:
    mode: DecodeContextMode = "dense"
    block_size: int = 1
    block_num: int | None = None
    prefix_block_num: int = 0
    random_seed: int = 42

    @property
    def is_dense(self) -> bool:
        return self.mode == "dense"

    def validate(self) -> None:
        valid_modes = {"dense", "recent", "uniform", "random", "prefix_recent"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported decode context mode: {self.mode}")
        if self.block_size <= 0:
            raise ValueError("decode_context_block_size must be positive.")
        if self.mode != "dense" and self.block_num is None:
            raise ValueError("decode_context_block_num is required for sparse decode modes.")
        if self.block_num is not None and self.block_num <= 0:
            raise ValueError("decode_context_block_num must be positive when set.")
        if self.prefix_block_num < 0:
            raise ValueError("decode_context_prefix_block_num must be non-negative.")


def config_from_sampling_params(
    sampling_params: SamplingParams,
    fallback: DecodeContextConfig,
) -> DecodeContextConfig:
    if sampling_params.decode_context_mode is None:
        return fallback

    return DecodeContextConfig(
        mode=sampling_params.decode_context_mode,
        block_size=(
            fallback.block_size
            if sampling_params.decode_context_block_size is None
            else sampling_params.decode_context_block_size
        ),
        block_num=sampling_params.decode_context_block_num,
        prefix_block_num=(
            fallback.prefix_block_num
            if sampling_params.decode_context_prefix_block_num is None
            else sampling_params.decode_context_prefix_block_num
        ),
        random_seed=(
            fallback.random_seed
            if sampling_params.decode_context_random_seed is None
            else sampling_params.decode_context_random_seed
        ),
    )


def select_decode_positions(seq_len: int, req_uid: int, config: DecodeContextConfig) -> list[int]:
    """Return sorted logical KV positions selected for one decode request."""
    config.validate()
    if seq_len <= 0:
        return []
    if config.is_dense:
        return list(range(seq_len))

    assert config.block_num is not None
    block_size = config.block_size
    num_blocks = (seq_len + block_size - 1) // block_size
    keep_blocks = min(config.block_num, num_blocks)

    if keep_blocks >= num_blocks:
        selected_blocks = list(range(num_blocks))
    elif config.mode == "recent":
        selected_blocks = list(range(num_blocks - keep_blocks, num_blocks))
    elif config.mode == "uniform":
        selected_blocks = _select_uniform_blocks(num_blocks, keep_blocks)
    elif config.mode == "random":
        selected_blocks = _select_random_blocks(num_blocks, keep_blocks, req_uid, seq_len, config)
    elif config.mode == "prefix_recent":
        prefix = min(config.prefix_block_num, keep_blocks, num_blocks)
        recent = keep_blocks - prefix
        selected = set(range(prefix))
        selected.update(range(max(prefix, num_blocks - recent), num_blocks))
        if len(selected) < keep_blocks:
            selected.update(range(prefix, min(num_blocks, prefix + keep_blocks - len(selected))))
        selected_blocks = sorted(selected)
    else:
        raise ValueError(f"Unsupported decode context mode: {config.mode}")

    return _expand_blocks(selected_blocks, block_size, seq_len)


def _select_uniform_blocks(num_blocks: int, keep_blocks: int) -> list[int]:
    if keep_blocks == 1:
        return [num_blocks - 1]
    selected: list[int] = []
    used: set[int] = set()
    for i in range(keep_blocks):
        block = round(i * (num_blocks - 1) / (keep_blocks - 1))
        if block not in used:
            selected.append(block)
            used.add(block)
    block = num_blocks - 1
    while len(selected) < keep_blocks:
        if block not in used:
            selected.append(block)
            used.add(block)
        block -= 1
    return sorted(selected)


def _select_random_blocks(
    num_blocks: int,
    keep_blocks: int,
    req_uid: int,
    seq_len: int,
    config: DecodeContextConfig,
) -> list[int]:
    if keep_blocks == 1:
        return [num_blocks - 1]
    rng = random.Random(config.random_seed + req_uid * 1_000_003 + seq_len)
    older_blocks = list(range(num_blocks - 1))
    selected = rng.sample(older_blocks, keep_blocks - 1)
    selected.append(num_blocks - 1)
    return sorted(selected)


def _expand_blocks(blocks: list[int], block_size: int, seq_len: int) -> list[int]:
    positions: list[int] = []
    for block in blocks:
        start = block * block_size
        end = min(start + block_size, seq_len)
        positions.extend(range(start, end))
    return positions
