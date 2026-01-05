import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym  # Use Gymnasium instead of OpenAI Gym
from gymnasium import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
import logging
import json
import matplotlib.pyplot as plt
from collections import deque
import time
import matplotlib
matplotlib.use("Agg")
SEED = 0  

import random
import numpy as np
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Detailed logging
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Hugging Face model name
MIN_PROMPT_LENGTH = 3                   # Minimum allowed prompt length
MAX_PROMPT_LENGTH = 7                   # Maximum allowed prompt length
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # Directory for saving results
CHECKPOINT_FILE = os.path.join(SAVE_PATH, "checkpoint.json")
POPULATION_SIZE = 20                    # Population size for evolutionary search
GENERATIONS = 200                       # Total number of generations to run
MUTATION_RATE = 0.05                    # Probability of mutating each token
ALPHA = 2                               # Factor to penalize prompt length in reward
TP_THRESHOLDS = (512, 1024)             # Tail-Persistence thresholds


def choose_device() -> str:
    """
    Pick a safe device:
      - use CPU if CUDA is not available
      - use CPU if the GPU has a compute capability beyond what this PyTorch
        build is likely compiled for (e.g., Blackwell sm_120, which gave
        'no kernel image is available' errors in the logs).
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        return "cpu"

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        logger.info("CUDA device: %s (capability %d.%d)", name, major, minor)

        if major * 10 + minor > 90:
            logger.warning(
                "GPU capability %d.%d is likely unsupported by this PyTorch build; "
                "using CPU instead to avoid 'no kernel image' errors.",
                major, minor,
            )
            return "cpu"
    except Exception as e:
        logger.warning(
            "Could not query CUDA device capability (%s); falling back to CPU.", e
        )
        return "cpu"

    return "cuda"


DEVICE = choose_device()



# ---------------------------
# LOAD Llama-2-7b-hf MODEL & TOKENIZER
# ---------------------------
logger.info("Loading Llama-2-7b-hf model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Use float16 on CUDA, bfloat16 or float32 on CPU to avoid weird behaviour
dtype = torch.float16 if DEVICE == "cuda" else torch.float32

llama_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
).to(DEVICE)
CONTEXT_WINDOW =  llama_model.config.max_position_embeddings               # nominal window for Llama-2-7b-hf; adjust per run
logger.info("Llama-2-7b-hf model loaded on %s with dtype %s", DEVICE, dtype)

MAX_OUTPUT_TOKENS = 4 * CONTEXT_WINDOW              # Target maximum output tokens from generation
# Enable gradient checkpointing for memory efficiency
llama_model.gradient_checkpointing_enable()
logger.debug("Gradient checkpointing enabled for Llama-2-7b-hf model")

EOS_TOKEN_ID = tokenizer.eos_token_id
VOCAB_SIZE = tokenizer.vocab_size
logger.info("EOS_TOKEN_ID: %d, VOCAB_SIZE: %d", EOS_TOKEN_ID, VOCAB_SIZE)

# For the all-token ablation we allow (almost) the full vocabulary,
# excluding just EOS and PAD to avoid trivial termination / padding tokens.
EXCLUDED_TOKENS = {EOS_TOKEN_ID, tokenizer.pad_token_id}

ALLOWED_TOKEN_IDS = [
    tid for tid in range(tokenizer.vocab_size)
    if tid not in EXCLUDED_TOKENS
]

if not ALLOWED_TOKEN_IDS:
    raise ValueError("No valid token IDs found for search space; check tokenizer and config.")

logger.info("Configuration: " + json.dumps({
    "MODEL_NAME": MODEL_NAME,
    "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
    "MIN_PROMPT_LENGTH": MIN_PROMPT_LENGTH,
    "MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH,
    "DEVICE": DEVICE,
    "SAVE_PATH": SAVE_PATH,
    "POPULATION_SIZE": POPULATION_SIZE,
    "GENERATIONS": GENERATIONS,
    "MUTATION_RATE": MUTATION_RATE,
    "ALPHA": ALPHA,
}))

DECODING = {
    "do_sample": True,
    "temperature": None,    # set if you vary
    "top_p": None,          # set if you vary
    "repetition_penalty": None,
    "stop_sequences": None,
    "max_new_tokens": MAX_OUTPUT_TOKENS,
}
logger.info("Decoding/Server knobs: " + json.dumps(DECODING))

os.makedirs(SAVE_PATH, exist_ok=True)
logger.info("Save path ensured at: %s", SAVE_PATH)

PRE_EVALS = int(os.environ.get("PRE_EVALS", "0"))

# Enable TF32 only when running on CUDA
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.debug("TF32 enabled for CUDA matmul")
else:
    logger.debug("Running on CPU; TF32 CUDA matmul not applicable.")
# -------- Dedup helpers --------
def compute_tail_persistence(new_tokens, k_gram=8):
    """Cheap novelty proxy: measure tail tokens after the last novel k-gram."""
    if len(new_tokens) < k_gram * 2:
        return {f"TP@{n}": 0 for n in TP_THRESHOLDS}, 0
    seen = set()
    last_novel_idx = -1
    for i in range(len(new_tokens) - k_gram + 1):
        gram = tuple(new_tokens[i:i+k_gram])
        if gram not in seen:
            seen.add(gram)
            last_novel_idx = i + k_gram - 1
    tail_len = max(0, len(new_tokens) - (last_novel_idx + 1))
    metrics = {f"TP@{n}": int(tail_len >= n) for n in TP_THRESHOLDS}
    return metrics, tail_len

def attack_side_metrics(new_tokens, context_window, saw_eos, cap_hit, latency_sec=None, cost_usd=None):
    ogf = len(new_tokens) / max(1, context_window)
    stall = int((not saw_eos) and cap_hit)
    tp_flags, tail_len = compute_tail_persistence(new_tokens)
    out = {"OGF": ogf, "stall": stall, "tail_len": tail_len, **tp_flags}
    if latency_sec is not None: out["latency_sec"] = latency_sec
    if cost_usd is not None: out["cost_usd"] = cost_usd
    return out
    
def _normalize_prompt_text(text: str) -> str:
    # Collapse whitespace so "hello   world" == "hello world"
    return " ".join(text.split())

def _load_seen_prompts(file_path: str) -> set[str]:
    """
    Parse your existing 'prompts_over_8192_llama.txt' file and collect all
    previously saved prompts into a set (normalized) so we don't resave them.
    """
    seen = set()
    if not os.path.exists(file_path):
        return seen
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            buf = f.read()
        # Your file writes blocks like:
        #  --- Prompt that generated >8192 tokens ---
        #  Prompt:
        #  <prompt text>
        #  Number of tokens generated: ...
        blocks = buf.split("--- Prompt that generated >8192 tokens ---")
        for block in blocks:
            if "Prompt:" not in block:
                continue
            # Extract text between "Prompt:\n" and the next metadata line
            part = block.split("Prompt:\n", 1)[1]
            prompt_text = part.split("\nNumber of tokens generated", 1)[0].strip()
            if prompt_text:
                seen.add(_normalize_prompt_text(prompt_text))
    except Exception as e:
        logger.warning(f"Failed to parse existing prompts file for dedup: {e}")
    return seen

# ---------------------------
# EVOLUTIONARY SEARCH FUNCTIONS
# ---------------------------
def evaluate_prompt(prompt_tokens):
    inputs = torch.tensor(prompt_tokens).unsqueeze(0).to(DEVICE)
    gen_kwargs = {
        "max_new_tokens": MAX_OUTPUT_TOKENS,
        "do_sample": bool(DECODING.get("do_sample", True)),
        "temperature": DECODING.get("temperature"),
        "top_p": DECODING.get("top_p"),
        "repetition_penalty": DECODING.get("repetition_penalty"),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    t0 = time.time()
    try:
        with torch.no_grad():
            gen_ids = llama_model.generate(inputs, **gen_kwargs)
    except RuntimeError as e:
        msg = str(e)
        # Fatal: architecture / kernel mismatch 
        if "no kernel image is available for execution on the device" in msg:
            logger.error(
                "[FATAL] Generation failed due to GPU kernel incompatibility: %s\n"
                "This likely means the GPU architecture is not supported by the "
                "current PyTorch build. Aborting run so we do NOT save corrupt "
                "checkpoints.",
                msg,
            )
            raise  # let PBS see a non-zero exit
        # Optional: keep a 'soft' fallback for other, genuinely per-prompt errors
        logger.warning(
            "[Fallback] Prompt caused a non-fatal generation failure: %s "
            "(returning large negative reward).",
            msg,
        )
        return -1e6, [], {
            "OGF": 0.0,
            "stall": 0,
            "tail_len": 0,
            "TP@512": 0,
            "TP@1024": 0,
            "latency_sec": None,
        }


    latency = time.time() - t0
    new_tokens = gen_ids[0, inputs.shape[1]:].tolist() 
    gen_length = len(new_tokens)

    reward = gen_length - ALPHA * len(prompt_tokens)

    # EOS penalty on tail
    saw_eos = (EOS_TOKEN_ID in new_tokens)
    if saw_eos:
        eos_position = new_tokens.index(EOS_TOKEN_ID)
        penalty = 5 * (MAX_OUTPUT_TOKENS - eos_position)
        reward -= penalty

    cap_hit = (gen_length >= MAX_OUTPUT_TOKENS)
    m = attack_side_metrics(
        new_tokens,
        context_window=CONTEXT_WINDOW,
        saw_eos=saw_eos,
        cap_hit=cap_hit,
        latency_sec=latency,
        cost_usd=None
    )
    return reward, new_tokens, m




def mutate_prompt(prompt_tokens, mutation_rate=MUTATION_RATE):
    """Mutate a prompt using only allowed token IDs."""
    new_prompt = prompt_tokens.copy()
    for i in range(len(new_prompt)):
        if random.random() < mutation_rate:
            new_prompt[i] = random.choice(ALLOWED_TOKEN_IDS)
    return new_prompt


def crossover_prompts(prompt1, prompt2):
    """Perform a one-point crossover between two prompts."""
    length = len(prompt1)
    if length < 2:
        return prompt1  # or some fallback
    crossover_point = random.randint(1, length - 1)
    new_prompt = prompt1[:crossover_point] + prompt2[crossover_point:]
    return new_prompt

def save_checkpoint(generation, population, best_prompt, best_reward):
    """Saves the current state to a checkpoint file."""
    checkpoint = {
        "generation": generation,
        "population": population,
        "best_prompt": best_prompt,
        "best_reward": best_reward
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    logger.info("Checkpoint saved at generation %d", generation)

def load_checkpoint():
    """Loads the checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        logger.info("Loaded checkpoint from generation %d", checkpoint["generation"])
        return (checkpoint["generation"], checkpoint["population"],
                checkpoint["best_prompt"], checkpoint["best_reward"])
    else:
        return None

# ---------------------------
# EVOLUTIONARY SEARCH WITH VARIABLE PROMPT LENGTH & CHECKPOINTING
# ---------------------------
def evolutionary_search_variable():
    """
    Evolves a population of prompts with variable lengths between MIN_PROMPT_LENGTH
    and MAX_PROMPT_LENGTH. Rewards favor longer outputs while penalizing longer prompts.
    We also:
      - Save any prompt that generates more than 8192 tokens.
      - Track the best prompt that creates exactly 16384 tokens.
    Checkpointing is implemented to allow continuation of training.
    Returns the best prompt, best reward, and lists of generation metrics.
    """
    # Prepare files for saving large-output prompts and the 16384-token best
    import os
    prompts_over_8192_file = os.path.join(SAVE_PATH, "prompts_over_8192_llama.txt")
    best_prompt_16384_file = os.path.join(SAVE_PATH, "best_prompt_16384_llama.txt")
    # Dedup: load any already-saved prompts so we never save duplicates
    seen_prompts_over_8192 = _load_seen_prompts(prompts_over_8192_file)
    logger.info("Loaded %d previously saved >8192 prompts for deduplication.", len(seen_prompts_over_8192))

    # Track the best prompt that generates exactly 16384 tokens
    best_16384_prompt = None
    best_16384_reward = -float("inf")

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        start_gen, population, best_prompt, best_reward = checkpoint
        start_gen += 1  # resume from next generation
    else:
        start_gen = 0
        # Initialize population
        initial_prompt_text = "In"
        base_prompt_tokens = [
            tok for tok in tokenizer.encode(initial_prompt_text, add_special_tokens=False)
            if tok in ALLOWED_TOKEN_IDS
        ]
        population = []
        for _ in range(POPULATION_SIZE):
            rand_length = random.randint(MIN_PROMPT_LENGTH, MAX_PROMPT_LENGTH)
            if len(base_prompt_tokens) < rand_length:
                candidate = base_prompt_tokens + [
                    random.choice(ALLOWED_TOKEN_IDS)
                    for _ in range(rand_length - len(base_prompt_tokens))
                ]
            else:
                candidate = base_prompt_tokens[:rand_length]
            population.append(candidate)
        best_prompt = None
        best_reward = -float("inf")

    # Lists to store metrics for visualization
    best_rewards_per_gen = []
    avg_rewards_per_gen = []

    # create CSV once
    EVAL_CSV = os.path.join(SAVE_PATH, "ea_eval_log.csv")
    if not os.path.exists(EVAL_CSV):
        with open(EVAL_CSV, "w") as f:
            f.write("gen,idx,prompt_len,produced,OGF,stall,TP@512,TP@1024,tail_len,latency_sec,"
                    "model,temperature,top_p,rep_penalty,stop,max_new_tokens\n")
    
    success_ogf2 = 0
    success_ogf4 = 0
    total_evals = 0
    
    for gen in range(start_gen, GENERATIONS):
        logger.info("Generation %d START", gen)
        evaluated = []

        
        for i, prompt in enumerate(population):
            reward, generated, m = evaluate_prompt(prompt)
            evaluated.append((reward, prompt, generated, m))
            
            # success counters
            total_evals += 1
            success_ogf2 += int(m["OGF"] >= 2.0)
            success_ogf4 += int(m["OGF"] >= 4.0)
            
            # CSV row
            with open(EVAL_CSV, "a") as f:
                f.write(
                    f"{gen},{i},{len(prompt)},{len(generated)},"
                    f"{(m.get('OGF') or 0.0):.4f},{m.get('stall',0)},"
                    f"{m.get('TP@512',0)},{m.get('TP@1024',0)},{m.get('tail_len',0)},"
                    f"{(m.get('latency_sec') or 0.0):.4f},"
                    f"{MODEL_NAME},"
                    f"{'' if DECODING.get('temperature') is None else DECODING['temperature']},"
                    f"{'' if DECODING.get('top_p') is None else DECODING['top_p']},"
                    f"{'' if DECODING.get('repetition_penalty') is None else DECODING['repetition_penalty']},"
                    f"{bool(DECODING.get('stop_sequences') or [])},"
                    f"{DECODING.get('max_new_tokens',0)}\n"
                )
            
                                    
            gen_length = len(generated)
            log_msg = (
                f"[Gen {gen} | Timestep {i}] Prompt Length: {len(prompt)} "
                f"| Generated Tokens: {gen_length} | Reward: {reward}"
            )
            logger.debug(log_msg)
            print(log_msg)

            # Update best overall prompt
            if reward > best_reward:
                best_reward = reward
                best_prompt = prompt
                with open(os.path.join(SAVE_PATH, "best_prompt.txt"), "w") as f:
                    f.write(tokenizer.decode(best_prompt, skip_special_tokens=True))

            # (1) Save any prompt that generates > 8192 tokens (no duplicates)
            if gen_length > 8192:
                decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
                key = _normalize_prompt_text(decoded_prompt)
                tok_strings = [tokenizer.decode([tid], skip_special_tokens=True) for tid in prompt]
                if key not in seen_prompts_over_8192:
                    with open(prompts_over_8192_file, "a", encoding="utf-8") as f:
                        f.write("\n--- Prompt that generated >8192 tokens ---\n")
                        f.write("\nToken strings: " + repr(tok_strings) + "\n")
                        f.write("Prompt:\n")
                        f.write(decoded_prompt)
                        f.write("\nNumber of tokens generated: " + str(gen_length))
                        f.write("\nReward: " + str(reward))
                        f.write("\n------------------------------------------\n")
                    seen_prompts_over_8192.add(key)
                    logger.debug("Saved new >8192 prompt (unique).")
                else:
                    logger.debug("Skipped saving duplicate >8192 prompt.")


            # (2) Track the best prompt that exactly generates 16384 tokens
            if gen_length == 16384 and reward > best_16384_reward:
                best_16384_reward = reward
                best_16384_prompt = prompt
                with open(best_prompt_16384_file, "w") as f:
                    f.write("Best prompt that generated exactly 16384 tokens:\n")
                    f.write(tokenizer.decode(best_16384_prompt, skip_special_tokens=True))
                    f.write("\nReward: " + str(best_16384_reward) + "\n")
                    
        succ2_rate = success_ogf2 / max(1, total_evals)
        succ4_rate = success_ogf4 / max(1, total_evals)
        display_evals = PRE_EVALS + total_evals  # shift x-axis by pre-metrics calls
        logger.info(
            f"[Budgeted Success] after {display_evals} evals (offset {PRE_EVALS}): "
            f"Success@OGF≥2={succ2_rate:.3f}, Success@OGF≥4={succ4_rate:.3f}"
        )
        
        # Compute generation metrics
        rewards = [ev[0] for ev in evaluated]
        gen_avg = sum(rewards) / len(rewards)
        best_rewards_per_gen.append(best_reward)
        avg_rewards_per_gen.append(gen_avg)

        # If absolutely every reward is the failure sentinel, treat as fatal
        if all(r <= -9e5 for r in rewards):
            logger.error(
                "All rewards in generation %d are failure sentinels (<= -9e5). "
                "This suggests a systemic issue (e.g., device failure). "
                "Aborting instead of saving a corrupt checkpoint.",
                gen,
            )
            raise RuntimeError(
                f"All evaluations failed in generation {gen}; see log for details."
            )

        # Generation summary
        gen_summary = {
            "generation": gen,
            "population": [
                {"prompt_length": len(item[1]), "reward": item[0]}
                for item in evaluated
            ],
            "best_reward": best_reward,
            "average_reward": gen_avg,
        }
        summary_msg = f"Generation {gen} SUMMARY: {json.dumps(gen_summary)}"
        logger.info(summary_msg)
        print(summary_msg)

        # Save checkpoint at the end of each generation
        save_checkpoint(gen, population, best_prompt, best_reward)

        # ---- at the end of the per-generation loop (after logging "[Budgeted Success]" and any checkpoint) ----
        if gen % 10 == 0:  # auto-save a plot every 10 generations
            try:
                import pandas as pd, numpy as np
                import matplotlib
                matplotlib.use("Agg")  # headless-safe
                import matplotlib.pyplot as plt
                import os
        
                csv_path = os.path.join(SAVE_PATH, "ea_eval_log.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if len(df) >= 1:
                        # one row == one eval, in chronological order
                        df["cum_evals"] = np.arange(1, len(df) + 1)
                        df["display_evals"] = PRE_EVALS + df["cum_evals"]
        
                        df["succ2"] = (df.get("OGF", 0) >= 2).astype(int)
                        df["succ4"] = (df.get("OGF", 0) >= 4).astype(int)
                        df["rate2"] = df["succ2"].cumsum() / df["cum_evals"]
                        df["rate4"] = df["succ4"].cumsum() / df["cum_evals"]
        
                        plt.figure(figsize=(7, 4))
                        plt.plot(df["display_evals"], df["rate2"], label="Success@OGF≥2")
                        plt.plot(df["display_evals"], df["rate4"], label="Success@OGF≥4")
                        plt.axvline(PRE_EVALS, linestyle="--", linewidth=1, label="metrics enabled")
                        plt.xlabel("Total evaluations (including pre-metrics offset)")
                        plt.ylabel("Cumulative success rate")
                        plt.legend()
                        plt.tight_layout()
        
                        out_png = os.path.join(SAVE_PATH, f"budget_curve_gen{gen}.png")
                        plt.savefig(out_png, dpi=150)
                        plt.close()
                        logger.info("Saved budget curve: %s", out_png)
            except Exception as e:
                logger.warning("Plotting failed at gen %d: %s", gen, e)

        # Selection: Keep top half of population based on reward
        evaluated.sort(key=lambda x: x[0], reverse=True)
        survivors = [item[1] for item in evaluated[:POPULATION_SIZE // 2]]

        new_population = survivors.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(survivors, 2)
            child = crossover_prompts(parent1, parent2)
            child = mutate_prompt(child, MUTATION_RATE)
            # Enforce that child length remains between MIN_PROMPT_LENGTH and MAX_PROMPT_LENGTH
            if len(child) < MIN_PROMPT_LENGTH:
                child += [
                    random.choice(ALLOWED_TOKEN_IDS)
                    for _ in range(MIN_PROMPT_LENGTH - len(child))
                ]
            if len(child) > MAX_PROMPT_LENGTH:
                child = child[:MAX_PROMPT_LENGTH]
            new_population.append(child)
        population = new_population

        print(f"--- End of Generation {gen} ---\n")

    logger.info("Evolutionary search completed. Best reward: %d", best_reward)
    return (best_prompt, best_reward,
            best_rewards_per_gen, avg_rewards_per_gen,
            best_16384_prompt, best_16384_reward)

# ---------------------------
# RUN THE EVOLUTIONARY SEARCH
# ---------------------------
results = evolutionary_search_variable()
best_prompt, best_reward, best_rewards, avg_rewards, best_16384_prompt, best_16384_reward = results

logger.info(
    "Best prompt found (reward %d): %s",
    best_reward,
    tokenizer.decode(best_prompt, skip_special_tokens=True)
)
print("Best prompt found:", tokenizer.decode(best_prompt, skip_special_tokens=True))

if best_16384_prompt is not None:
    logger.info(
        "Best prompt that created 16384 tokens (reward %d): %s",
        best_16384_reward,
        tokenizer.decode(best_16384_prompt, skip_special_tokens=True)
    )
    print(
        "Best prompt that created 16384 tokens:",
        tokenizer.decode(best_16384_prompt, skip_special_tokens=True)
    )

# ---------------------------
# PLOT REWARDS
# ---------------------------
generations = list(range(len(best_rewards)))
plt.figure(figsize=(10, 5))
plt.plot(generations, best_rewards, label='Best Reward', marker='o')
plt.plot(generations, avg_rewards, label='Average Reward', marker='x')
plt.xlabel("Generation")
plt.ylabel("Reward")
plt.title("Evolutionary Search Reward per Generation")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_PATH, "reward_progress.png"))


# ---------------------------
# TEST BEST PROMPT
# ---------------------------
def test_best_prompt(prompt_tokens):
    """Test the best prompt; print both continuation-only and full text."""
    inputs = torch.tensor(prompt_tokens).unsqueeze(0).to(DEVICE)
    gen_kwargs = {
        "max_new_tokens": MAX_OUTPUT_TOKENS,
        "do_sample": bool(DECODING.get("do_sample", True)),
        "temperature": DECODING.get("temperature"),
        "top_p": DECODING.get("top_p"),
        "repetition_penalty": DECODING.get("repetition_penalty"),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    with torch.no_grad():
        out = llama_model.generate(
            inputs, **gen_kwargs
        )
    # NEW: slice to get only the newly generated tokens
    new_tokens = out[0, inputs.shape[1]:].tolist()
    full_tokens = prompt_tokens + new_tokens

    decoded_continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded_full = tokenizer.decode(full_tokens, skip_special_tokens=True)

    logger.info("Test (continuation only):\n%s", decoded_continuation)
    print("Test (continuation only):\n", decoded_continuation)

    logger.info("Test (prompt + continuation):\n%s", decoded_full)
    print("Test (prompt + continuation):\n", decoded_full)

    # (Optional) sanity check on characters
    if not all(c.isalnum() or c.isspace() for c in decoded_continuation):
        logger.warning("Continuation contains non-alnum/space chars.")
    return decoded_continuation


# Finally, test the best overall prompt:
test_best_prompt(best_prompt)

