"""Isolation eval for the Planner: zero-shot base vs fine-tuned adapter.

Loads the E4B base + LoRA adapter ONCE, then for each eval row predicts the
category twice — adapter ON (fine-tuned) and ``disable_adapter()`` OFF (the bare
zero-shot base) — and scores both against the gold category. This is the paper's
controlled comparison ("does fine-tuning help?") on a HELD-OUT set the model
never trained on (build the eval JSONL from training/problems + eval_labels).

Reports overall accuracy, per-category accuracy, and a confusion matrix for each
mode. Mirrors infer_planner's multimodal tokenize path (render with
tokenize=False, then the underlying text tokenizer).

Run (jaehwan-ml):
    docker exec -i -w /workspace -e CUDA_VISIBLE_DEVICES=0 jaehwan-ml \
        python -m training.eval_planner --data training/sft/heldout_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from unsloth import FastModel

HERE = Path(__file__).resolve().parent
CATEGORIES = ("FORM-based", "MCS", "Advanced")


def parse_category(text: str) -> str:
    """Extract the predicted category from a generation (JSON or substring)."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        cat = obj.get("recommended_category", "")
        if cat in CATEGORIES:
            return cat
    except Exception:
        pass
    # Fallback: first category string that appears. Check "FORM-based" before
    # bare tokens; "MCS"/"Advanced" are unambiguous.
    for cat in CATEGORIES:
        if cat in text:
            return cat
    return "?"


def generate(model, tok, prompt_msgs, tokenizer) -> str:
    text = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to("cuda") for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    return tok.decode(out[0][plen:], skip_special_tokens=True).strip()


def score(rows, preds) -> dict:
    correct = sum(p == g for g, p in preds)
    per_cat = {c: [0, 0] for c in CATEGORIES}     # [correct, total]
    confusion = Counter()                          # (gold, pred)
    for gold, pred in preds:
        per_cat[gold][1] += 1
        per_cat[gold][0] += (pred == gold)
        confusion[(gold, pred)] += 1
    return {"acc": correct / len(preds), "per_cat": per_cat, "confusion": confusion}


def report(name: str, res: dict) -> None:
    print(f"\n===== {name} =====")
    print(f"overall accuracy: {res['acc']*100:.1f}%")
    for c in CATEGORIES:
        cor, tot = res["per_cat"][c]
        print(f"  {c:11s}: {cor}/{tot}" + (f" ({cor/tot*100:.0f}%)" if tot else ""))
    print("  confusion (gold -> pred):")
    for c in CATEGORIES:
        row = "  ".join(f"{p}:{res['confusion'].get((c, p), 0)}" for p in CATEGORIES)
        misc = res["confusion"].get((c, "?"), 0)
        print(f"    {c:11s} -> {row}" + (f"  ?:{misc}" if misc else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(HERE / "adapters" / "planner"))
    ap.add_argument("--data", default=str(HERE / "sft" / "heldout_eval.jsonl"))
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--out", default=None,
                    help="If set, write structured results JSON here (results/...).")
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter, max_seq_length=args.max_seq_len,
        load_in_4bit=True, attn_implementation="eager",
    )
    try:
        FastModel.for_inference(model)
    except Exception:
        pass
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(rows)} held-out eval rows from {args.data}")

    ft_preds, base_preds = [], []
    for row in rows:
        msgs = row["messages"]
        prompt_msgs, gold_json = msgs[:-1], msgs[-1]["content"]
        gold = json.loads(gold_json)["recommended_category"]

        ft = parse_category(generate(model, tok, prompt_msgs, tokenizer))     # adapter ON
        with model.disable_adapter():
            base = parse_category(generate(model, tok, prompt_msgs, tokenizer))  # zero-shot
        ft_preds.append((gold, ft))
        base_preds.append((gold, base))
        print(f"  {gold:11s} | zero-shot={base:11s} | fine-tuned={ft:11s}"
              + ("  <-- FT fixed" if base != gold and ft == gold else "")
              + ("  <-- FT broke" if base == gold and ft != gold else ""))

    base_res, ft_res = score(rows, base_preds), score(rows, ft_preds)
    report("ZERO-SHOT base E4B", base_res)
    report("FINE-TUNED E4B", ft_res)

    if args.out:
        def entry(name, trained, preds, res):
            return {"name": name, "engine": "unsloth", "trained": trained,
                    "overall": res["acc"],
                    "per_class": {c: res["per_cat"][c] for c in CATEGORIES},
                    "predictions": [{"gold": g, "pred": p} for g, p in preds]}
        out = {"eval_set": args.data, "n": len(rows),
               "entries": [entry("E4B zero-shot", False, base_preds, base_res),
                           entry("E4B fine-tuned", True, ft_preds, ft_res)]}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
