---
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- transformers
- trl
- unsloth
---

# Planner LoRA adapter

LoRA adapter for the Planner agent of *Automating Structural Reliability
Analysis with a Multi-agent Large Language Model Framework*. Given a
structured reliability problem specification (`ProblemSpec`), the Planner
returns one analysis category: FORM-based, MCS or Advanced.

## Weights

`adapter_model.safetensors` is a release asset: download it from
[v1.0.0](/JaehwanJeon/Reliability-LLM-supplementary-material/releases/download/v1.0.0/adapter_model.safetensors)
into this `planner/` folder.

## Base model

`unsloth/gemma-4-E4B-it` (the snapshot recorded in `adapter_config.json`, and
the `--base` default in `../source_scripts/training/train_planner.py`).

## Input and output

The Planner was trained on `[system, user, assistant]` chat records
(`../../01_training_records_150/planner_gen_sft.jsonl`). All records share one
system prompt. The user turn is the `ProblemSpec` as indented JSON between a
fixed prefix and suffix:

```text
Problem spec:
[spec]
{ ...ProblemSpec JSON... }

Classify this problem into one analysis category.
```

The benchmark specifications in
`../../02_benchmark_inputs_20/current_revision/problems/` use the same
`ProblemSpec` format. The training target is a compact JSON object:

```json
{"recommended_category":"FORM-based"}
```

The value is one of `"FORM-based"`, `"MCS"` or `"Advanced"`.
