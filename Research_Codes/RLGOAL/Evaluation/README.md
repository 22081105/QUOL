# Evaluation, tables, and plots (RL-GOAL)

This folder contains three scripts that support the **evaluation pipeline** used in the paper:

1) `infer_rl_goal.py` — run RL-GOAL (or a no-prefix baseline) against a victim model and log per-run metrics to CSV  
2) `make_rlgoal.py` — merge chunk CSVs and produce paper-style **summary tables** (CSV)  
3) `plot_rlgoal.py` — load the same CSVs and produce the paper-style **figures** (PDF/PNG)

---

## 0) Expected inputs/outputs

### Input CSVs
`infer_rl_goal.py` produces chunked CSVs (one per job/chunk), e.g.

- `policy_llama_chunk00_of_15.csv` … `policy_llama_chunk14_of_15.csv`
- `baseline_noprefix_llama_chunk00_of_15.csv` … `baseline_noprefix_llama_chunk14_of_15.csv`

### Output tables (CSV)
`make_rlgoal.py` writes:

- `rlgoal_results_main.csv`
- `rlgoal_results_appendix.csv`
- `rlgoal_results_latency.csv`

### Output figures
`plot_rlgoal.py` writes plots

---

## 1) Running evaluation: `infer_rl_goal.py`

### What it does
- Loads a trained RL-GOAL checkpoint (`--ckpt`) **or** runs a no-prefix baseline (`--random_policy --max_steps 0`).
- Executes `--num_samples` independent runs and logs per-run metrics (OGF, stall, latency, etc.) to `--csv`.

### Chunked runs (recommended on HPC)
Use `--chunk_index` and `--num_chunks` so each job evaluates a disjoint slice of runs.

**Important:** when `--num_chunks > 1`, the script automatically rewrites `--csv` to a chunk-specific file name:
`<root>_chunkXX_of_YY.csv` (avoids collisions and is array-job safe).

### Common commands

**(A) RL-GOAL checkpoint evaluation (example; 15 chunks)**
```bash
python infer_rl_goal.py   --ckpt ./rl_goal/YOUR_CHECKPOINT.pt   --victim_model meta-llama/Llama-2-7b-hf   --reward_mode victim   --goal 16384   --max_steps 64   --num_samples 1000   --seed 0   --csv policy_llama.csv   --chunk_index $PBS_ARRAY_INDEX   --num_chunks 15
```

**(B) No-prefix baseline (example; 15 chunks)**
```bash
python infer_rl_goal.py   --random_policy   --max_steps 0   --num_samples 1000   --victim_model meta-llama/Llama-2-7b-hf   --reward_mode victim   --goal 16384   --seed 42   --csv baseline_noprefix_llama.csv   --chunk_index $PBS_ARRAY_INDEX   --num_chunks 15
```

### Key flags (practical)
- `--victim_model`: Hugging Face model ID used as the victim.
- `--goal`: generation budget target (often `B = 4C`, e.g., `16384` for `C=4096`).
- `--max_steps`: how many tokens RL-GOAL may emit (attacker prefix length cap).
- `--temp`, `--top_p`, `--top_k`: sampling settings for victim generation.
- `--context_window`: if unset, derived from victim model config and used for OGF computation.
- `--gate_alpha`: optional EOS gating factor (0 disables gating).

### CSV schema (columns)
Each row contains run identity + metrics. The script writes these columns:

`run, base_seed, chunk_index, num_chunks, ckpt, victim_model, goal, max_steps, temp, top_p, top_k, gate_alpha, reward_mode, context_window, torch_version, transformers_version, gen_len, OGF, stall, tail_len, TP@512, TP@1024, latency_sec, success_ogf1, success_ogf2, success_ogf4, reward, validity, eos_pen, ppl`

---

## 2) Building tables: `make_rlgoal.py`

### What it does
- Loads all chunk CSVs per condition (policy vs no-prefix baseline).
- Produces three CSV files suitable to copy into LaTeX tables (mean±std + rates).

### Run it
Point `--dir` at the directory that contains the chunk CSVs.

```bash
python make_rlgoal.py   --dir .   --n_chunks 15   --out_main rlgoal_results_main.csv   --out_appendix rlgoal_results_appendix.csv   --out_latency rlgoal_results_latency.csv
```

### If you changed file names or number of chunks
- Update `--n_chunks`
- If your chunk filename templates differ from the defaults, edit the `policy_groups` and `baseline_groups`
  lists inside `make_rlgoal.py`.

---

## 3) Making plots: `plot_rlgoal.py`

### What it does
Loads the same evaluation CSVs and writes the three paper-style figures.

### Run it
```bash
python plot_rlgoal.py   --in_dir .   --out_dir ./plots   --make_rlgoal ./make_rlgoal.py
```

Optional filters:
- `--exclude_tags ds` (exclude e.g. DeepSeek)
- `--include_tags llama,phi3` (allowlist)

Example:
```bash
python plot_rlgoal.py   --in_dir .   --out_dir ./plots   --make_rlgoal ./make_rlgoal.py   --exclude_tags ds
```

---

## Notes for artifact/reproducibility
- Prefer chunked evaluation (array jobs) so reruns are simple and deterministic.
- Keep `--goal`, sampling parameters, and `--max_steps` consistent with what is reported in the paper.
- The scripts only use **relative** output paths by default (artifact-friendly).
