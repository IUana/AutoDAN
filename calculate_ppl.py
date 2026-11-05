"""Calculate the AutoDAN paper's GPT-2 sentence-perplexity metric.

The paper states that Table 1 uses standard Sentence PPL evaluated by GPT-2.
The released repository does not contain the corresponding evaluation script.
This implementation evaluates each final, expanded jailbreak prompt separately
and reports the arithmetic mean of those sentence-level PPL values.
"""

import argparse
import json
import math
from pathlib import Path


PAPER_MODEL = "gpt2"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/Models/gpt2"
PAPER_TORCH_VERSION = "2.7.1"
PAPER_TRANSFORMERS_VERSION = "4.56.2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="AutoDAN result JSON")
    parser.add_argument("--output", default="ppl_results.json")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.items() if isinstance(data, dict) else enumerate(data)
    records = []
    for sample_id, item in entries:
        if isinstance(item, list):
            if len(item) != 1:
                raise ValueError(
                    f"sample {sample_id!r} contains {len(item)} responses; "
                    "Table 1 requires exactly one final prompt per behavior"
                )
            item = item[0]
        if not isinstance(item, dict):
            raise TypeError(f"sample {sample_id!r} is not a JSON object")
        records.append((str(sample_id), item))
    return records


def expanded_prompt(item):
    field = "final_suffix" if "final_suffix" in item else "prompt"
    if field not in item or "goal" not in item:
        raise KeyError("each sample must contain goal and final_suffix (or prompt)")

    # This exactly follows autodan_SuffixManager.get_prompt() in the official code.
    return item[field].replace("[REPLACE]", item["goal"].lower()), field


def main():
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_version = torch.__version__.split("+")[0]
    if torch_version != PAPER_TORCH_VERSION:
        raise RuntimeError(
            f"strict mode requires torch=={PAPER_TORCH_VERSION}, found {torch.__version__}"
        )
    if transformers.__version__ != PAPER_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "strict mode requires transformers=="
            f"{PAPER_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )

    input_path = Path(args.input)
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"local GPT-2 directory not found: {model_path}")
    records = load_records(input_path)
    if not records:
        raise ValueError("input contains no samples")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()

    results = []
    prompt_fields = set()
    with torch.inference_mode():
        for sample_id, item in records:
            text, prompt_field = expanded_prompt(item)
            prompt_fields.add(prompt_field)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            token_count = input_ids.shape[1]
            if token_count < 2:
                raise ValueError(f"sample {sample_id!r} has fewer than two GPT-2 tokens")
            if token_count > model.config.n_positions:
                raise ValueError(
                    f"sample {sample_id!r} has {token_count} tokens, exceeding GPT-2's "
                    f"{model.config.n_positions}-token context. The paper does not disclose "
                    "a long-text windowing rule, so this script will not invent one."
                )

            mean_nll = model(input_ids=input_ids, labels=input_ids).loss.item()
            results.append(
                {
                    "id": sample_id,
                    "predicted_tokens": token_count - 1,
                    "mean_nll": mean_nll,
                    "sentence_ppl": math.exp(mean_nll),
                }
            )

    values = [row["sentence_ppl"] for row in results]
    report = {
        "metric": "mean sentence perplexity",
        "model": PAPER_MODEL,
        "model_path": str(model_path.resolve()),
        "local_files_only": True,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision_reported_by_config": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "prompt_preprocessing": "replace [REPLACE] with goal.lower()",
        "prompt_fields_seen": sorted(prompt_fields),
        "aggregation": "arithmetic mean of per-sample sentence PPL",
        "sample_count": len(results),
        "mean_sentence_ppl": math.fsum(values) / len(values),
        "items": results,
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"samples: {report['sample_count']}")
    print(f"mean sentence PPL: {report['mean_sentence_ppl']:.4f}")
    print(f"details: {output_path.resolve()}")


if __name__ == "__main__":
    main()
