from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal online sparse decode benchmark.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--model", default=None, help="Model name. Defaults to server model.")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="sparse_decode_results.csv")
    parser.add_argument("--jsonl", default="sparse_decode_raw.jsonl")

    # Labels only. The server must be launched with matching --decode-context-* flags.
    parser.add_argument("--label", default="")
    parser.add_argument("--decode-context-mode", default="unknown")
    parser.add_argument("--decode-context-block-size", type=int, default=-1)
    parser.add_argument("--decode-context-block-num", type=int, default=-1)
    parser.add_argument("--decode-context-prefix-block-num", type=int, default=0)
    parser.add_argument("--decode-context-random-seed", type=int, default=-1)
    return parser.parse_args()


def generate_prompt(tokenizer: Any, n: int) -> str:
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]
    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            token_ids.extend(random.randint(0, vocab_size) for _ in range(n - len(token_ids)))
        else:
            token_ids = token_ids[:n]
    raise ValueError(f"Failed to generate a prompt with {n} tokens.")


async def get_model_name(client: Any) -> str:
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available from server.")


async def benchmark_one(
    client: Any,
    prompt: str,
    output_len: int,
    model: str,
    extra_body: Dict[str, Any],
) -> List[float]:
    response = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_len,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1, **extra_body},
    )
    tics = [time.perf_counter()]
    async for _ in response:
        tics.append(time.perf_counter())
    return tics


async def benchmark_batch(
    client: Any,
    prompts: List[str],
    output_len: int,
    model: str,
    extra_body: Dict[str, Any],
) -> List[List[float]]:
    tasks = [benchmark_one(client, prompt, output_len, model, extra_body) for prompt in prompts]
    return await asyncio.gather(*tasks)


def percentile(values: List[float], q: float) -> float:
    assert values
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * q))
    return values[idx]


def summarize(
    raw_results: List[List[float]],
    started_at: float,
    finished_at: float,
) -> Dict[str, float]:
    ttfts: List[float] = []
    tpots: List[float] = []
    e2es: List[float] = []
    generated_tokens = 0

    for tics in raw_results:
        if len(tics) < 2:
            continue
        deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
        ttfts.append(deltas[0])
        tpots.extend(deltas[1:])
        e2es.append(tics[-1] - tics[0])
        generated_tokens += len(tics) - 1

    duration = max(finished_at - started_at, 1e-9)
    summary: Dict[str, float] = {
        "duration_s": duration,
        "generated_tokens": float(generated_tokens),
        "request_throughput_req_s": len(raw_results) / duration,
        "token_throughput_tok_s": generated_tokens / duration,
    }
    for name, values in [("ttft", ttfts), ("tpot", tpots), ("e2e", e2es)]:
        if values:
            summary[f"{name}_avg_ms"] = mean(values) * 1000
            summary[f"{name}_p50_ms"] = percentile(values, 0.50) * 1000
            summary[f"{name}_p90_ms"] = percentile(values, 0.90) * 1000
            summary[f"{name}_p99_ms"] = percentile(values, 0.99) * 1000
        else:
            summary[f"{name}_avg_ms"] = 0.0
            summary[f"{name}_p50_ms"] = 0.0
            summary[f"{name}_p90_ms"] = 0.0
            summary[f"{name}_p99_ms"] = 0.0
    return summary


def append_csv(path: str, row: Dict[str, Any]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def make_decode_context_body(args: argparse.Namespace) -> Dict[str, Any]:
    if args.decode_context_mode == "unknown":
        return {}
    body: Dict[str, Any] = {"decode_context_mode": args.decode_context_mode}
    if args.decode_context_block_size > 0:
        body["decode_context_block_size"] = args.decode_context_block_size
    if args.decode_context_block_num > 0:
        body["decode_context_block_num"] = args.decode_context_block_num
    if args.decode_context_prefix_block_num > 0:
        body["decode_context_prefix_block_num"] = args.decode_context_prefix_block_num
    if args.decode_context_random_seed >= 0:
        body["decode_context_random_seed"] = args.decode_context_random_seed
    return body


async def main() -> None:
    args = parse_args()
    from openai import AsyncOpenAI as OpenAI
    from transformers import AutoTokenizer

    random.seed(args.seed)
    extra_body = make_decode_context_body(args)

    base_url = f"http://{args.host}:{args.port}/v1"
    async with OpenAI(base_url=base_url, api_key="EMPTY") as client:
        model = args.model or await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt = generate_prompt(tokenizer, args.input_len)
        prompts = [prompt] * args.num_requests

        if args.warmup > 0:
            warmup_prompts = [prompt] * args.warmup
            await benchmark_batch(client, warmup_prompts, args.output_len, model, extra_body)

        started_at = time.perf_counter()
        raw_results = await benchmark_batch(client, prompts, args.output_len, model, extra_body)
        finished_at = time.perf_counter()

    summary = summarize(raw_results, started_at, finished_at)
    row: Dict[str, Any] = {
        "label": args.label,
        "model": model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_requests": args.num_requests,
        "decode_context_mode": args.decode_context_mode,
        "decode_context_block_size": args.decode_context_block_size,
        "decode_context_block_num": args.decode_context_block_num,
        "decode_context_prefix_block_num": args.decode_context_prefix_block_num,
        "decode_context_random_seed": args.decode_context_random_seed,
        **summary,
    }
    append_csv(args.csv, row)

    raw_records = [
        {
            **row,
            "request_id": i,
            "tics": tics,
            "actual_output_chunks": len(tics) - 1,
        }
        for i, tics in enumerate(raw_results)
    ]
    append_jsonl(args.jsonl, raw_records)

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
