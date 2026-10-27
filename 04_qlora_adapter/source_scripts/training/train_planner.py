"""QLoRA fine-tune the Planner: (problem spec -> category) classifier.

Trains a LoRA adapter on ``training/sft/planner_gen_sft.jsonl`` (chat-format pairs
built by ``training.build_sft``). The base is Gemma-4-E4B (the locked planner
model); because Gemma-4 is multimodal we load it through unsloth's ``FastModel``
(NOT ``FastLanguageModel``) and tune only the language layers.

Key Gemma-4 specifics (verified 2026-06-11, not the usual Gemma-3 defaults):
  * loader      : ``FastModel`` with ``attn_implementation="eager"`` (unsloth
                  warns Gemma-4 is "strongly recommended" to train with eager).
  * turn markers: ``<|turn>{role}`` / ``<turn|>`` — NOT ``<start_of_turn>``.
                  ``train_on_responses_only`` is masked on these so loss lands
                  ONLY on the ``<|turn>model`` (assistant) turn.

The base is a CLI arg (``--base``) so a different Gemma-4 variant can be swapped
in without editing code. Output is a LoRA adapter only; merging + GGUF convert +
``ollama create gemma4-e4b`` (replacing the served tag in place) is a separate,
later step.

Run (in the ml container):
    docker exec -i -w /workspace jaehwan-ml python -m training.train_planner
"""

from __future__ import annotations

import argparse
from pathlib import Path

# unsloth MUST be imported before transformers/trl so its patches apply.
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "sft" / "planner_gen_sft.jsonl"
DEFAULT_OUT = HERE / "adapters" / "planner"

# Gemma-4 turn markers (verified against the gemma-4-31B-it tokenizer template).
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="unsloth/gemma-4-E4B-it",
                   help="Base model (HF id or local path). unsloth maps this to "
                        "its prequantized 4bit checkpoint.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="SFT JSONL (chat 'messages' format).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Where to save the LoRA adapter.")
    # 4096 OOMs on a single 24GB A5000 (base+optim ~19GB + long-seq activations).
    # 2048 fits (~19GB); drops the 3 longest samples (>2048 tok). To keep all 35 at
    # long context, shorten the system prompt or train on 2 GPUs.
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data} (run: python -m training.build_sft)")

    # --- model ---------------------------------------------------------------
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,           # QLoRA: quantize the frozen base on the fly
        full_finetuning=False,
        attn_implementation="eager",  # Gemma-4 recommendation
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        # planner is text-only: tune language layers, leave the vision tower frozen.
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # --- data ----------------------------------------------------------------
    # Each row already has chat 'messages' [system, user, assistant]; render it
    # with the model's own template so train format == serve format.
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def render(example: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = dataset.map(render, remove_columns=dataset.column_names)
    print(f"Loaded {len(dataset)} SFT examples from {args.data}")

    # --- trainer -------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(args.out / "_trainer"),
            report_to="none",
        ),
    )

    # Mask the prompt: compute loss ONLY on the assistant (<|turn>model) turn.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    trainer.train()

    # --- save adapter --------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
