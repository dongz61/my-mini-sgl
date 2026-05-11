from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke sweep for request-level sparse decode.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-dir", default="/root/data/ziqian/prompt")
    parser.add_argument("--prompt-template", default="synthetic_{k}k.txt")
    parser.add_argument("--context-lens", default="4096,8192")
    parser.add_argument("--patterns", default="recent,uniform")
    parser.add_argument("--block-sizes", default="64")
    parser.add_argument("--select-ratios", default="0.25,0.5")
    parser.add_argument("--prefix-block-num", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--csv", default="results/sparse_decode_smoke.csv")
    parser.add_argument("--jsonl", default="results/sparse_decode_smoke.jsonl")
    parser.add_argument("--raw-jsonl", default="results/sparse_decode_smoke_raw.jsonl")
    return parser.parse_args()


async def get_model_name(client: Any) -> str:
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available from server.")


def prompt_path(prompt_dir: str, template: str, context_len: int) -> Path:
    if context_len % 1024 != 0:
        raise ValueError(f"context_len must be K-aligned for default template: {context_len}")
    return Path(prompt_dir) / template.format(k=context_len // 1024, context_len=context_len)


def load_prompt(prompt_dir: str, template: str, context_len: int) -> str:
    path = prompt_path(prompt_dir, template, context_len)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()


def build_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    context_lens = parse_int_list(args.context_lens)
    patterns = parse_str_list(args.patterns)
    block_sizes = parse_int_list(args.block_sizes)
    select_ratios = parse_float_list(args.select_ratios)

    experiments: List[Dict[str, Any]] = []
    for context_len in context_lens:
        if args.include_dense:
            experiments.append(
                {
                    "experiment_type": "native_dense",
                    "context_len": context_len,
                    "sparse_pattern": None,
                    "block_size": None,
                    "total_blocks": None,
                    "select_ratio_target": 1.0,
                    "block_num": None,
                    "prefix_block_num": None,
                    "random_seed": None,
                    "selected_tokens": context_len,
                    "select_ratio_actual": 1.0,
                    "request_body": {"decode_context_mode": "dense"},
                }
            )

        for pattern in patterns:
            for block_size in block_sizes:
                total_blocks = math.ceil(context_len / block_size)
                for ratio in select_ratios:
                    block_num = max(1, round(total_blocks * ratio))
                    selected_tokens = min(context_len, block_num * block_size)
                    experiments.append(
                        {
                            "experiment_type": "sparse",
                            "context_len": context_len,
                            "sparse_pattern": pattern,
                            "block_size": block_size,
                            "total_blocks": total_blocks,
                            "select_ratio_target": ratio,
                            "block_num": block_num,
                            "prefix_block_num": (
                                args.prefix_block_num if pattern == "prefix_recent" else None
                            ),
                            "random_seed": args.random_seed if pattern == "random" else None,
                            "selected_tokens": selected_tokens,
                            "select_ratio_actual": selected_tokens / context_len,
                            "request_body": {
                                "decode_context_mode": pattern,
                                "decode_context_block_size": block_size,
                                "decode_context_block_num": block_num,
                                **(
                                    {"decode_context_prefix_block_num": args.prefix_block_num}
                                    if pattern == "prefix_recent"
                                    else {}
                                ),
                                **(
                                    {"decode_context_random_seed": args.random_seed}
                                    if pattern == "random"
                                    else {}
                                ),
                            },
                        }
                    )

    return experiments[: args.max_runs]


async def request_one(client: Any, model: str, prompt: str, output_len: int, body: Dict[str, Any]) -> List[float]:
    response = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_len,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1, **body},
    )
    tics = [time.perf_counter()]
    async for _ in response:
        tics.append(time.perf_counter())
    return tics


async def run_requests(
    client: Any,
    model: str,
    prompt: str,
    output_len: int,
    body: Dict[str, Any],
    num_prompts: int,
    concurrency: int,
) -> List[List[float]]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded() -> List[float]:
        async with sem:
            return await request_one(client, model, prompt, output_len, body)

    return await asyncio.gather(*[guarded() for _ in range(num_prompts)])


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def summarize(tic_lists: List[List[float]], started_at: float, finished_at: float) -> Dict[str, float]:
    ttfts: List[float] = []
    itls: List[float] = []
    e2es: List[float] = []
    output_chunks = 0
    for tics in tic_lists:
        if len(tics) < 2:
            continue
        deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
        ttfts.append(deltas[0])
        itls.extend(deltas[1:])
        e2es.append(tics[-1] - tics[0])
        output_chunks += len(tics) - 1

    duration = max(finished_at - started_at, 1e-9)
    result: Dict[str, float] = {
        "duration_s": duration,
        "output_chunks": float(output_chunks),
        "output_tokens_per_sec": output_chunks / duration,
        "request_throughput_req_s": len(tic_lists) / duration,
    }
    for name, values in (("ttft", ttfts), ("itl", itls), ("e2e", e2es)):
        if values:
            result[f"{name}_ms_avg"] = mean(values) * 1000
            result[f"{name}_ms_p50"] = percentile(values, 0.50) * 1000
            result[f"{name}_ms_p90"] = percentile(values, 0.90) * 1000
            result[f"{name}_ms_p99"] = percentile(values, 0.99) * 1000
        else:
            result[f"{name}_ms_avg"] = 0.0
            result[f"{name}_ms_p50"] = 0.0
            result[f"{name}_ms_p90"] = 0.0
            result[f"{name}_ms_p99"] = 0.0
    result["tpot_ms_avg"] = result["itl_ms_avg"]
    return result


def append_csv(path: str, row: Dict[str, Any]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


async def main() -> None:
    args = parse_args()
    from openai import AsyncOpenAI as OpenAI

    experiments = build_experiments(args)
    base_url = f"http://{args.host}:{args.port}/v1"
    async with OpenAI(base_url=base_url, api_key="EMPTY") as client:
        model = args.model or await get_model_name(client)

        for run_id, exp in enumerate(experiments):
            prompt = load_prompt(args.prompt_dir, args.prompt_template, exp["context_len"])
            if args.warmup > 0:
                await run_requests(
                    client,
                    model,
                    prompt,
                    args.output_len,
                    exp["request_body"],
                    args.warmup,
                    args.concurrency,
                )

            started_at = time.perf_counter()
            tic_lists = await run_requests(
                client,
                model,
                prompt,
                args.output_len,
                exp["request_body"],
                args.num_prompts,
                args.concurrency,
            )
            finished_at = time.perf_counter()

            summary = summarize(tic_lists, started_at, finished_at)
            row = {
                "run_id": run_id,
                "model": model,
                "output_len": args.output_len,
                "num_prompts": args.num_prompts,
                "concurrency": args.concurrency,
                "success": True,
                "error": None,
                **{k: v for k, v in exp.items() if k != "request_body"},
                **summary,
            }
            append_csv(args.csv, row)
            append_jsonl(args.jsonl, [row])
            append_jsonl(
                args.raw_jsonl,
                (
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "context_len": exp["context_len"],
                        "experiment_type": exp["experiment_type"],
                        "sparse_pattern": exp["sparse_pattern"],
                        "block_size": exp["block_size"],
                        "block_num": exp["block_num"],
                        "tics": tics,
                        "output_chunks": len(tics) - 1,
                    }
                    for request_id, tics in enumerate(tic_lists)
                ),
            )
            print(json.dumps(row, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
