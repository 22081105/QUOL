# EOGen Evaluation

This folder contains the **evaluation harness** used in the paper to measure over-generation for:
- **EOGen (prefix prompts)** (`test_eogen/`)
- **EOGen-suffix** (`suffix/`)
- **Baselines** (`baseline/`)
and to produce:
- **Combined plots** (`plots/`)
- **Latency summary tables** (`latency/`)

All commands below are written **relative to the repository root**.  

---

## 1) Directory layout

```
Research_Codes/EOGen/evaluation/
  deepseek-coder-7b-base-v1.5/
    baseline/
    suffix/
    test_eogen/
  Llama-2-7b-hf/
    baseline/
    suffix/
    test_eogen/
  Phi-3-mini-4k-instruct/
    baseline/
    suffix/
    test_eogen/
  plots/
    ea.py
  latency/
    make_latency_table.py
  make_table.py
```

Each `<MODEL>/...` folder is self-contained and writes outputs **inside that folder**.

---

## 2) Environment / dependencies

Python 3.10+ recommended.

Minimum runtime packages (typical):
- `torch`
- `transformers`
- `numpy`, `pandas`
- `matplotlib` (plots)

Requirement file added

---

## 3) Evaluation configuration (paper settings)

Across **prefix**, **suffix**, and **baselines**, use consistent victim generation settings unless stated:
- `do_sample=True`, `temperature=1.0`, `top_p=1.0`
- Context window `C` read from the model config.
- Budget: **`B = 4C`** new tokens (e.g., `C=4096` ⇒ `B=16384`).

Trials:
- **EOGen** and **EOGen-suffix**: **`T=10`** stochastic decodes per prompt.
- **Baselines**: controlled by `TRIALS_PER_PROMPT` inside `run_baselines.py`.

Stall:
- A run is marked as a **stall** if it hits the token budget `B` **without** producing an EOS token.

---

## 4) Running the evaluations (chunked / parallel / resumable)

All three settings are run in **15 chunks** by default (`chunk00_of_15` … `chunk14_of_15`).
You can run chunks sequentially or in parallel (e.g., HPC job arrays).

### 4.1 Prefix evaluation (EOGen prompts)

Run per model from that model’s `test_eogen/` folder.

Example (DeepSeek):

```bash
cd Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/test_eogen

# run one chunk
python3 test_eogen.py --chunk_index 0 --num_chunks 15 --csv eogen_results_ds.csv

# (optional) after all chunks exist, merge only
python3 test_eogen.py --merge_only 1 --num_chunks 15 --csv eogen_results_ds.csv
```

Outputs (DeepSeek naming):
- `eogen_results_ds_chunk00_of_15.csv` … `eogen_results_ds_chunk14_of_15.csv`
- merged: `eogen_results_ds_merged.csv`

For **Phi-3** and **LLaMA-2**, run the corresponding `test_eogen.py` in their folders with the appropriate `--csv` base name used by that folder (the script derives chunk file names from it).

> Note: the DeepSeek `test_eogen.py` expects the prompt file `prompts_over_8192_ds.txt` to be present in the working directory. If your prompt file name differs, either rename it or update the `PROMPT_FILE` constant in that script.

---

### 4.2 Baselines

Run per model from that model’s `baseline/` folder.

Example (DeepSeek):

```bash
cd Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/baseline

python3 run_baselines.py \
  --chunk_index 0 \
  --num_chunks 15 \
  --wizard_file wizardlm.jsonl \
  --output_csv baseline_results_ds.csv
```

Outputs:
- `baseline_results_ds_chunk00_of_15.csv` … `baseline_results_ds_chunk14_of_15.csv`

Baselines included (per `run_baselines.py`):
- Repeat-style (repeat/recursion/count/longtext/code)
- Infinite babble
- Random short (default 100 prompts)
- WizardLM-style instruction baseline (default 100 prompts, loaded from `wizardlm.jsonl`)

---

### 4.3 EOGen-suffix evaluation

Run per model from that model’s `suffix/` folder.

Example (DeepSeek):

```bash
cd Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/suffix

python3 suffix_output.py \
  --chunk_index 0 \
  --num_chunks 15 \
  --prompts prompts_over_8192_ds.txt \
  --input wizardlm.jsonl \
  --csv ds_suffix_results.csv \
  --output ds_result.jsonl
```

Outputs:
- `ds_suffix_results_chunk00_of_15.csv` … `ds_suffix_results_chunk14_of_15.csv`
- `ds_result_chunk00_of_15.jsonl` … (JSONL logs of generations)

---

## 5) Building the main over-generation table (EOGen / suffix / baselines)

`make_table.py` merges chunked CSVs and emits a compact table with columns:

`Prompt source | #Prompts | Avg. OGF | Succ. @≥1 | Succ. @≥2 | Succ. @≥3 | Succ. @≥4`

DeepSeek example (works regardless of where you run it because paths are explicit):

```bash
python3 Research_Codes/EOGen/evaluation/make_table.py \
  --num_chunks 15 \
  --eogen_dir   Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/test_eogen \
  --suffix_dir  Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/suffix \
  --baseline_dir Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/baseline \
  --eogen_prefix eogen_results_ds \
  --suffix_prefix ds_suffix_results \
  --baseline_prefix baseline_results_ds \
  --out_csv Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/ds_overgen_table.csv
```

For **Phi-3** and **LLaMA-2**, run the same command but:
1) point `--*_dir` to their model folders, and  
2) set `--*_prefix` to match the file basenames **before** `_chunkXX_of_15.csv`.

Tip to discover prefixes:
```bash
ls Research_Codes/EOGen/evaluation/Phi-3-mini-4k-instruct/test_eogen | head
```

---

## 6) Combined plots across 3 models

`plots/ea.py` overlays 3 models and produces:
- `eogen_success_hist_allmodels` (pdf+png)
- `eogen_maxogf_hist_allmodels` (pdf+png)
- `eogen_stall_cdf_allmodels` (pdf+png)

Run:

```bash
python3 Research_Codes/EOGen/evaluation/plots/ea.py \
  --phi_dir   Research_Codes/EOGen/evaluation/Phi-3-mini-4k-instruct/test_eogen \
  --llama_dir Research_Codes/EOGen/evaluation/Llama-2-7b-hf/test_eogen \
  --ds_dir    Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/test_eogen \
  --pattern   "eogen_results_*chunk*_of_15.csv" \
  --out_dir   Research_Codes/EOGen/evaluation/plots/out
```

Notes:
- Do **not** use a leading `/` on paths unless you truly mean an absolute path on Linux.
- `--pattern` defaults to `eogen_results_*chunk*_of_15.csv`, which matches the standard naming.

---

## 7) Latency summary tables

`latency/make_latency_table.py` reads the chunk CSVs under each model root and writes per-model latency tables.

Run:

```bash
python3 Research_Codes/EOGen/evaluation/latency/make_latency_table.py \
  --out_dir Research_Codes/EOGen/evaluation/latency/out \
  --phi_root Research_Codes/EOGen/evaluation/Phi-3-mini-4k-instruct \
  --ds_root  Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5 \
  --llama_root Research_Codes/EOGen/evaluation/Llama-2-7b-hf \
  --models phi3 ds llama \
  --n_chunks 15
```

Outputs:
- `latency/out/latency_table_phi3.csv`
- `latency/out/latency_table_ds.csv`
- `latency/out/latency_table_llama.csv`

Important detail (already handled in the script):
- For LLaMA baselines, it looks for `baseline_results_llama2_chunkXX_of_15.csv`.


