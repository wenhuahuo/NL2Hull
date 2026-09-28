#!/usr/bin/env python3
"""Evaluate a local Transformers causal model on direct FFD actions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_prompt_action_eval import (
    build_report,
    digest,
    evaluation_record,
    parse_json,
    prompt_for,
    score,
)


def evaluate_one(model: Any, tokenizer: Any, record: dict[str, Any], device: str) -> dict[str, Any]:
    result = {
        "task_key": record["task_key"],
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "input_text": record["text"],
        "target_actions": record["target_actions"],
    }
    try:
        prompt = prompt_for(record)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        prediction = parse_json(response)
        result.update({
            "status": "completed",
            "raw_response": response,
            "prediction": prediction,
            "score": score(prediction, record["target_actions"]),
        })
    except Exception as error:
        result.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = [
        evaluation_record(json.loads(line))
        for line in args.data.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError("empty evaluation dataset")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local Transformers evaluation")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "model": str(args.model),
        "record_count": len(records),
        "prompt_contract": "shared direct FFD action JSON prompt and score",
        "status": "running",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results = []
    with (args.output / "results.jsonl").open("w") as stream:
        for record in records:
            result = evaluate_one(model, tokenizer, record, "cuda")
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if len(results) % 25 == 0:
                print(f"evaluated {len(results)}/{len(records)}", flush=True)

    report = build_report(results, records, args.data, str(args.model))
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    completed = sum(result["status"] == "completed" for result in results)
    manifest.update({
        "status": "completed",
        "completed_records": completed,
        "failed_records": len(results) - completed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
