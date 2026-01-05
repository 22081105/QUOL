# EOGen (Evolutionary Over-Generation)

This repository contains the **EOGen** evolutionary search implementation used to discover short token-prefix prompts that induce **over-generation** in black-box / query-only settings.

The code is organized per victim model, with two LLaMA-2 ablation variants.

---

## Repository layout

> Paths shown with `/` separators; on Windows use `\`.

```
Research_Codes/
  EOGen/
    deepseek-coder-7b-base-v1.5/
      EA_DS.py
      ea_ds/                      # outputs (auto-created)
    Llama-2-7b-hf/
      EA_LLAMA.py
      ea_llama2/                  # outputs (auto-created)
      ablation1/
        EA_LLama1.py              # Ablation-1 (all-token search space)
        checkpoint.json           # optional
      ablation2/
        EA_LLama2.py              # Ablation-2 (English-word token space)
        checkpoint.json           # optional
    Phi-3-mini-4k-instruct/
      EA_PHI3.py
      ea_phi3/                    # outputs (auto-created)
```

Each script writes results to a **local output folder** next to the script:
- DeepSeek: `ea_ds/`
- LLaMA-2 main: `ea_llama2/`
- Phi-3: `ea_phi3/`
- Ablations: output is written into the **ablation folder itself** (because the ablation scripts live inside `ablation1/` and `ablation2/`).

---

## What the scripts do

All scripts implement an **evolutionary search** over short prompts:
- Prompt length: **3–7 tokens**
- Population size: **20**
- Mutation rate: **0.05 per token**
- Generations:
  - Main scripts: **500** for Llama-2-7b-hf and Phi-3-mini-4k-instruct, **387** for deepseek-coder-7b-base-v1.5
  - Ablations: **200** for ablation1 and **101** for ablation2 for Llama-2-7b-hf

Evaluation uses `transformers` `generate()` with a token budget:

- Nominal context window: `C = model.config.max_position_embeddings`
- Generation budget: `B = 4C` new tokens (e.g., for `C=4096`, `B=16384`)

### Reward (high level)
For each prompt evaluation, the reward is based on **produced length**, minus a **prompt-length penalty** (via `ALPHA`), and an **EOS penalty** that discourages early termination.

---

## Requirements

- Python **3.10+**
- CUDA-capable GPU recommended (scripts can fall back to CPU, but will be slow)

Python packages:
- `torch`
- `transformers`
- `gymnasium`
- `numpy`
- `pandas`
- `matplotlib`
- `nltk` (needed for the English-word token search space)

Example install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install torch transformers gymnasium numpy pandas matplotlib nltk
```

### NLTK corpus (offline-friendly)

Some scripts restrict the mutation/search space to tokens that decode into **ASCII alphabetic strings that appear in NLTK's English word list**.

Install the corpus once:

```bash
python -c "import nltk; nltk.download('words')"
```

If you are on an offline cluster, download on a machine with internet once, then copy the NLTK data directory, or set `NLTK_DATA` to a shared location.

---

## Model access (Hugging Face)

The scripts load models via:

- DeepSeek: `deepseek-ai/deepseek-coder-7b-base-v1.5`
- LLaMA-2: `meta-llama/Llama-2-7b-hf` (**gated**; requires HF access)
- Phi-3: `microsoft/Phi-3-mini-4k-instruct`

For gated models, authenticate with Hugging Face (one-time):

```bash
huggingface-cli login
```

Or set an access token in the environment:

```bash
export HUGGINGFACE_HUB_TOKEN=...
```

---

## Running the experiments

### 1) DeepSeek (EOGen)

```bash
cd Research_Codes/EOGen/deepseek-coder-7b-base-v1.5
python EA_DS.py
```

Outputs: `ea_ds/` (see “Outputs” below)

---

### 2) LLaMA-2 main (EOGen)

```bash
cd Research_Codes/EOGen/Llama-2-7b-hf
python EA_LLAMA.py
```

Outputs: `ea_llama2/`

---

### 3) Phi-3 (EOGen)

```bash
cd Research_Codes/EOGen/Phi-3-mini-4k-instruct
python EA_PHI3.py
```

Outputs: `ea_phi3/`

---

## Ablations (Llama-2-7b-hf)

### Ablation 1 — All-token search space
This variant allows (almost) the full vocabulary as the mutation space, excluding only EOS/PAD to avoid trivial termination/padding.

```bash
cd Research_Codes/EOGen/Llama-2-7b-hf/ablation1
python EA_LLama1.py
```

Key differences vs. main:
- `GENERATIONS = 200`
- `ALPHA = 0` (no prompt-length penalty)
- Token search space: **(almost) all tokens**, excluding EOS/PAD

Outputs: written into the `ablation1/` folder (next to the script), including `checkpoint.json` and plots/CSVs.

---

### Ablation 2 — English-word token search space
This variant restricts the mutation space to tokens that decode to ASCII alphabetic strings that occur in NLTK's English word list.

```bash
cd Research_Codes/EOGen/Llama-2-7b-hf/ablation2
python EA_LLama2.py
```

Key differences vs. main:
- `GENERATIONS = 101`
- `ALPHA = 0`
- Token search space: **English-word tokens only** (NLTK corpus required)

Outputs: written into the `ablation2/` folder.

---

## Resuming vs restarting (checkpointing)

Each run maintains a checkpoint file:

- `checkpoint.json`

If the checkpoint exists, the run resumes from the saved generation and population.
To restart from scratch, **delete** the checkpoint file in the output folder.

---

## Output files (per run)

Each run writes the following into its output folder (`SAVE_PATH`):

- `checkpoint.json`  
  Saved state for resuming runs: generation index, population, best prompt, best reward.

- `ea_eval_log.csv`  
  One row per evaluated prompt. Columns include:
  - `gen, idx, prompt_len`
  - `produced` (generated token count)
  - `OGF` (over-generation factor: `produced / C`)
  - `stall` (1 if cap hit without EOS)
  - `TP@512, TP@1024, tail_len` (tail-persistence metrics)
  - `latency_sec`
  - decoding/config fields (temperature/top_p/repetition_penalty/max_new_tokens, etc.)

- `best_prompt.txt`  
  Current best prompt (decoded text) according to reward.

- `prompts_over_8192_<model>.txt`  
  Deduplicated archive of prompts that produce very long outputs (threshold: >8192 tokens).  
  Useful for later evaluation / transfer tests.

- `best_prompt_16384_<model>.txt`  
  Best prompt among those that achieve exactly `16384` new tokens (when `B=16384`).

- `reward_progress.png`  
  Reward curve summary (best/avg reward vs. evals).

- `budget_curve_gen{gen}.png`  
  Per-generation “budgeted success” curves (when enabled in the script).

- `training.log` (or ablation-specific log name)  
  Run logs (configuration, progress, warnings, resume state, etc.)

---

## Optional knobs

### PRE_EVALS (x-axis offset for plots)
If you ran extra model queries **before** starting the evolutionary loop and you want plots/logs to reflect the total number of eval calls, set:

```bash
export PRE_EVALS=100   # example
```

By default it is `0`.

### Determinism / seeds
- Main scripts: set `SEED` in the script if you want fully reproducible randomness.
- Ablations: typically configured with a fixed seed already.

---

## Troubleshooting

- **No CUDA / GPU kernel errors**: the scripts can fall back to CPU, but performance will degrade significantly.
- **Gated LLaMA-2 model**: ensure your Hugging Face account has access and you have authenticated (`huggingface-cli login`).
- **NLTK corpus missing**: run `python -c "import nltk; nltk.download('words')"` once, or configure `NLTK_DATA`.

---

## Citation / attribution

If you use this code, cite the accompanying paper and explicitly reference:
- the model name
- context window `C` and budget `B=4C`
- population size, generations, and search-space restrictions (main vs. ablations)

---

## Evaluation and Results

The evaluation and results are in 
Research_Codes/
  EOGen/
    evaluation/
