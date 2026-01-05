# RL-GOAL (training code)

This directory contains the training-side implementation of **RL-GOAL**, a goal-conditioned PPO agent that constructs short **prompt tokens** to induce long generations from a **fixed, frozen victim LLM** (default: `meta-llama/Llama-2-7b-hf`). The victim model is **queried** for rollouts/rewards but is **never fine-tuned**.

## Files

- `RL_GOAL.py`  
  Main training script. Loads the victim LLM, defines the policy/value networks, runs PPO with replay + HER, logs diagnostics, and writes checkpoints.

- `llama_env.py`  
  Gymnasium environment `LLaMAMultiStepEnv-v0` used by `RL_GOAL.py`. Each step appends one token to the prompt; terminal reward is computed by calling the victim model to generate a continuation.

- `reward_tuning.py`  
  Reward + victim generation utilities (and some optional “reward tuning” helpers). Provides:
  - loading/tokenizer utilities
  - victim `generate()` call with a capped token budget
  - `compute_reward(...)` used by the environment/trainer

## Requirements

Python packages used by these scripts include:
- `torch`
- `transformers`
- `gymnasium`
- `numpy`
- `matplotlib`

A CUDA-capable GPU is strongly recommended; otherwise victim rollouts will be extremely slow.

## Model access / licensing notes

The default victim is `meta-llama/Llama-2-7b-hf`. You must have access to the model on Hugging Face and comply with the model’s license/terms.

## Quickstart (training)

From this directory:

```bash
# (1) Install deps in your environment (example; use your preferred tooling)
pip install torch transformers gymnasium numpy matplotlib

# (2) Run training
python RL_GOAL.py
```

By default, `RL_GOAL.py`:
1) runs a small number of “test episodes”, then  
2) starts the main training loop (long-running).

## Outputs (default paths)

The scripts write only **relative** paths (artifact-friendly):

- `logs/training_rl.log` — RL-GOAL training logs
- `logs/ppo_diagnostics.csv` — per-update PPO diagnostics
- `logs/reward_tuning.log` — reward/victim generation logs
- `rl_goal/trace.txt` — training trace output (if enabled in the script)
- `rl_goal/reward_curve_ep*.png` — periodic reward curves (saved every N episodes)

Checkpoints are saved under `./rl_goal/` (see `SAVE_PATH` in `RL_GOAL.py`).

## Key configuration knobs

### Victim decode budget / curriculum
The victim-side generation is capped via:
- `MAX_NEW_GEN` (victim generation budget cap)
- `VICTIM_BOOT_CAP` / internal boot cap (curriculum ramp)

Notes:
- `RL_GOAL.py` sets `MAX_NEW_GEN` to 16384 at import-time to match the evaluation convention `B=4C` (e.g., `C=4096 → B=16384`).
- The environment also computes an *effective* goal/budget based on `MAX_NEW_GEN` and the boot cap.

### Reward weights
In `reward_tuning.py`, reward weights can be controlled via environment variables:
- `W_LENGTH` (default 0.90)
- `W_PROMPT` (default 0.10)
- `W_EOS` (default 0.05)

### Policy positional limit / goal bucketing
In `RL_GOAL.py` (policy network), you can override:
- `POLICY_MAX_POS` (default 2048)
- `GOAL_BUCKET_SIZE` (default 256)

These affect the attacker policy/value transformer, not the victim model.

## Reproducibility

- Seeds are set (`SEED=0`) for Python/NumPy/PyTorch in both `RL_GOAL.py` and `reward_tuning.py`.
- Full determinism is not guaranteed due to GPU kernels and stochastic generation in the victim model.
- Training writes logs and checkpoints to support resumption and auditability.

## Changing the victim model

The victim model name is currently hard-coded in both:
- `RL_GOAL.py` (`MODEL_NAME = ...`)
- `reward_tuning.py` (`MODEL_NAME = ...`)

If you switch victims, update **both** to the same Hugging Face model ID (or refactor to read from an env var/CLI).

## Safety / responsible use

This code is intended for controlled robustness research. Run it only on models and infrastructure you are authorized to use, and follow applicable terms of service and institutional policies.

## Evaluation and Results
See Evaluation folder README for further details
