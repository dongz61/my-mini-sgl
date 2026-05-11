from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "python" / "minisgl" / "attention" / "decode_context.py"
spec = importlib.util.spec_from_file_location("decode_context", MODULE_PATH)
assert spec is not None and spec.loader is not None
decode_context = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = decode_context
spec.loader.exec_module(decode_context)

DecodeContextConfig = decode_context.DecodeContextConfig
select_decode_positions = decode_context.select_decode_positions


def test_recent_decode_context_selects_tail_blocks():
    config = DecodeContextConfig(mode="recent", block_size=4, block_num=2)

    assert select_decode_positions(seq_len=10, req_uid=0, config=config) == [4, 5, 6, 7, 8, 9]


def test_uniform_decode_context_includes_edges():
    config = DecodeContextConfig(mode="uniform", block_size=4, block_num=3)

    assert select_decode_positions(seq_len=20, req_uid=0, config=config) == [
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
    ]


def test_random_decode_context_is_reproducible_and_keeps_recent_block():
    config = DecodeContextConfig(mode="random", block_size=4, block_num=3, random_seed=7)

    first = select_decode_positions(seq_len=20, req_uid=2, config=config)
    second = select_decode_positions(seq_len=20, req_uid=2, config=config)

    assert first == second
    assert first[-4:] == [16, 17, 18, 19]
    assert len(first) == 12


def test_prefix_recent_decode_context_keeps_prefix_and_tail():
    config = DecodeContextConfig(
        mode="prefix_recent",
        block_size=4,
        block_num=3,
        prefix_block_num=1,
    )

    assert select_decode_positions(seq_len=20, req_uid=0, config=config) == [
        0,
        1,
        2,
        3,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]


def test_decode_context_rejects_invalid_config():
    config = DecodeContextConfig(mode="recent", block_size=0, block_num=1)

    with pytest.raises(ValueError):
        select_decode_positions(seq_len=10, req_uid=0, config=config)


def test_decode_context_requires_block_num_for_sparse_modes():
    config = DecodeContextConfig(mode="recent", block_size=4)

    with pytest.raises(ValueError):
        select_decode_positions(seq_len=10, req_uid=0, config=config)
