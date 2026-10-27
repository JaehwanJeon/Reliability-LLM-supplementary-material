# reliability-LLM — research materials

Supporting material for *Automating Structural Reliability Analysis with a
Multi-agent Large Language Model Framework*.

## Contents

| Folder | What it is |
|---|---|
| `01_training_records_150/` | The 150 chat-format SFT records used to fine-tune the Planner (`planner_gen_sft.jsonl`, 50 FORM-based / 50 MCS / 50 Advanced), plus an ID mapping (`id_mapping_787ff95.json`) to the generated problem IDs `g001`–`g150`. |
| `02_benchmark_inputs_20/` | The 20 natural-language benchmark prompts (`problems_nl.jsonl`) and, in `current_revision/`, the structured specifications and category labels in their current revision version. |
| `03_solver_code/` | Pure-Python reliability solvers (FORM, SORM, CLS, SuS, CE, iCE, SIS, MCS, IS), the spec-to-distribution converter and schemas, with `LICENSE` and `requirements.txt`. `reliability_llm/solvers/_era/` contains vendored ERA Group (TU München) sources under their own licence notice. |
| `04_qlora_adapter/` | The Planner LoRA adapter with its tokenizer, chat template and adapter card (`planner/README.md`), and the source training and evaluation scripts. |

## Data format

- **`planner_gen_sft.jsonl`**: one `{"messages": [system, user, assistant]}`
  record per line. All records share one system prompt. The user turn is
  `Problem spec:\n[spec]\n`, the `ProblemSpec` as indented JSON, then
  `\n\nClassify this problem into one analysis category.` The assistant turn
  is `{"recommended_category":"FORM-based"}` (or `"MCS"`, `"Advanced"`).
- **`id_mapping_787ff95.json`**: maps each record's position (`rows`) to its
  generated problem ID; the records themselves carry no ID.
- **`problems_nl.jsonl`**: one `{"id", "nl"}` record per line.
- **`current_revision/problems/*.json`**: `id`, `description`, `source`,
  `spec` and `meta`. `spec` holds `random_variables` (`name`, `distribution`,
  `mean`, `std`, `description`, `extra_params`), `limit_state_expr` and
  `correlation.matrix`.
- **`current_revision/eval_labels/*.json`**: true/false flags for
  `FORM-based`, `MCS` and `Advanced`, one file per problem with the same name.

The p11 and p17 specifications differ from the version used for the originally
reported Planner results.

## Planner adapter

The Planner takes a structured `ProblemSpec` and returns one analysis category:
FORM-based, MCS or Advanced. The adapter is a QLoRA fine-tune of
`unsloth/gemma-4-E4B-it`; see
[`04_qlora_adapter/planner/README.md`](04_qlora_adapter/planner/README.md).
