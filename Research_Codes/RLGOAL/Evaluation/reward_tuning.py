import os
import json
import random
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LogitsProcessor
import numpy as np
from collections import deque
# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------

logger = logging.getLogger("reward_logger")  # use a unique name
logger.setLevel(logging.DEBUG)
logger.propagate = False  # <--- prevents duplicate logs

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/reward_tuning.log", mode="a", delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_flush():
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()

# ---------------------------
# CONFIGURATION
# ---------------------------
SIMULATE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


# Put most weight on hitting long generations; de-emphasise prompt quality.
w_length = float(os.environ.get("W_LENGTH", 0.90))
w_prompt = float(os.environ.get("W_PROMPT", 0.10))
w_eos    = float(os.environ.get("W_EOS",    0.05))


ROLL_WINDOW = 50
recent_rewards        = deque(maxlen=ROLL_WINDOW)
recent_validities     = deque(maxlen=ROLL_WINDOW)
recent_eos_penalties  = deque(maxlen=ROLL_WINDOW)

SEED=0
ADJUST_INTERVAL = 10
TARGET_PROMPT_VALIDITY = 0.3
TARGET_GEN_LENGTH = 8192
LR_ADJUST = 0.02
NUM_EPISODES = 500000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# LOAD MODELS
# ---------------------------
tokenizer   = None
llama_model = None

coherence_tokenizer = None
coherence_model     = None
def set_models(_tok, _llm, _coh_tok=None, _coh_mod=None):
    global tokenizer, llama_model
    tokenizer           = _tok
    llama_model         = _llm


def _ensure_coherence_model():
    global coherence_tokenizer, coherence_model
    if coherence_model is None:
        logger.info("Loading DeBERTa coherence classifier …")
        coherence_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA")
        coherence_tokenizer.padding_side = "left"
        coherence_model = (AutoModelForSequenceClassification
                           .from_pretrained("textattack/bert-base-uncased-CoLA")
                           .to(DEVICE)
                           .eval())
    else:
        logger.debug("Coherence model is already loaded and in use.")
        
class ClampLogitsProcessor(LogitsProcessor):
    """
    Replace any nan/inf in the raw logits with large finite values
    so that softmax + sampling never sees invalid probabilities.
    """
    def __call__(self, input_ids, logits: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nan_to_num(logits,
                               nan=0.0,
                               posinf=20.0,
                               neginf=-20.0)

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        _ensure_model()  # ensure tokenizer is loaded
    return tokenizer

def _ensure_model():
    global tokenizer, llama_model
    if SIMULATE:
        return                    # dummy mode – nothing to do
    if llama_model is None:       # load only once – ever
        logger.info("Reward‑tuning: loading LLaMA weights …")
        tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
        llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",        # spread layers across all visible GPUs
            low_cpu_mem_usage=True,    # avoids full CPU weight copy
        ).eval()
        
        if DEVICE == "cpu":           # safety‑net
            logger.warning("Reward‑model loaded on CPU – evaluate_generation will be slow.")

def compute_prompt_validity(tokens: list[int]) -> float:
    """
    Coherence/fluency scoring removed. We return 1.0 to indicate 'valid'.
    Keep this function so env/trainer interfaces remain unchanged.
    """
    try:
        # still decode to ensure tokens are decodable, but ignore content quality
        _ = get_tokenizer().decode(tokens, skip_special_tokens=True)
        return 1.0
    except Exception:
        return 1.0  # be permissive

# -------------------------------------------------
MAX_GEN_LIMIT = 16384
MAX_NEW_GEN = int(os.environ.get("MAX_NEW_GEN", 4096))

def set_max_new_gen(n: int):
    global MAX_NEW_GEN
    try:
        MAX_NEW_GEN = min(int(n), MAX_GEN_LIMIT)
    except Exception:
        MAX_NEW_GEN = 4096

# ────────────────────────────────────────────────────────────
# DELETE the current evaluate_generation() definition
# ────────────────────────────────────────────────────────────
import math

def _calculate_perplexity(token_ids):
    """
    Compute (or estimate) perplexity on the given *already generated* ids.
    Replace the body with your model's native PPL if available.
    """
    if not token_ids:
        return 99.9

    # If you already have a proper PPL routine, call it here.
    # The simple heuristic below keeps the interface working.
    unique = max(1, len(set(token_ids)))
    ratio = len(token_ids) / float(unique)
    # lower is better; keep within a sane range
    est = 50.0 / max(1.0, ratio)
    return max(1.0, min(100.0, est))

# --- Victim boot cap (auto-rampable) ---
BOOT_CAP_MAX = 16384
_BOOT_CAP = int(os.getenv("VICTIM_BOOT_CAP", "2048"))  # start; env var wins if present

def set_boot_cap(n: int):
    """Set victim boot cap (monotonic in trainer)."""
    global _BOOT_CAP
    _BOOT_CAP = int(max(1, min(n, BOOT_CAP_MAX)))

def get_boot_cap() -> int:
    return int(_BOOT_CAP)

def evaluate_generation(prompt_tokens, goal_len=None):
    """
    Generate a continuation and return
       gen_length – number of new tokens
       eos_penalty
       perplexity – computed on the generated continuation
    """
    _ensure_model()

    if SIMULATE:
        gen_length   = random.randint(50, 150)
        eos_penalty  = random.uniform(5, 20) if random.random() < 0.3 else 0
        pseudo_ppl   = random.uniform(5.0, 15.0)
        return gen_length, eos_penalty, pseudo_ppl

    if not prompt_tokens:
        raise ValueError("evaluate_generation() received an empty prompt.")

    boot_cap    = get_boot_cap()
    allowed_new = min(int(goal_len), MAX_NEW_GEN, boot_cap)

    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    prompt_len = int(len(prompt_tokens))

    try:
        output = llama_model.generate(
            prompt_ids,
            max_new_tokens=allowed_new,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[ClampLogitsProcessor()],
            return_dict_in_generate=True,
        )
        logger.info(f"[victim] generate done: new_tokens={output.sequences.shape[1] - prompt_len}")
    except Exception as e:
        logger.error("[GEN ERROR] %s", e, exc_info=True)
        return 0, 10, 99.9

    sequences  = output.sequences[0].tolist()     # full prompt+continuation
    gen_ids    = sequences[prompt_len:]           # continuation only
    gen_length = len(gen_ids)

    # ----- EOS penalty ------------------------------------
    if tokenizer.eos_token_id in gen_ids:
        first_eos_offset = gen_ids.index(tokenizer.eos_token_id)
        eos_penalty      = 0.0 if first_eos_offset >= (goal_len or 0) * 0.9 else 5.0
    else:
        eos_penalty = 0.0

    perplexity = _calculate_perplexity(gen_ids) if gen_ids else 99.9
    if not math.isfinite(perplexity):
        perplexity = 1.0e6

    return gen_length, eos_penalty, perplexity

def compute_reward(
    prompt_tokens,
    goal_len=None,
    agent_gen_ids=None,
    used_eos=False,
):
    """
    Returns:
        reward, gen_length, prompt_validity, eos_penalty, perplexity
    Behavior:
        - If agent_gen_ids is provided, reward is computed on the *policy's* output.
        - Otherwise falls back to evaluate_generation(prompt_tokens, goal_len).
    """
    # ---------------- choose which tokens to score ----------------
    if agent_gen_ids is not None:
        # strip obvious special ids if you use 0/1/2 for <pad>/<bos>/<eos>
        gen_ids = [int(t) for t in agent_gen_ids if int(t) > 2]
        gen_length = len(gen_ids)

        # Early-EOS penalty relative to requested goal
        eos_penalty = 0.0
        if used_eos and goal_len:
            if gen_length < int(0.90 * int(goal_len)):
                eos_penalty = 5.0

        perplexity = _calculate_perplexity(gen_ids) if gen_ids else 99.9
    else:
        # Fallback path (kept for tools/ablation): free-run LLaMA
        gen_length, eos_penalty, perplexity = evaluate_generation(prompt_tokens, goal_len)

    # ---------------- compute reward terms ----------------
    prompt_validity = compute_prompt_validity(prompt_tokens)

    target_len = TARGET_GEN_LENGTH          # 8192 by default
    raw_goal   = int(goal_len) if goal_len is not None else target_len

    # Always treat "desired length" as at least target_len,
    # but never above MAX_NEW_GEN.
    eff_goal = float(max(target_len, min(raw_goal, MAX_NEW_GEN)))

    # Normalised length ratio relative to eff_goal (>= 8k)
    length_ratio = gen_length / eff_goal

    # Strongly penalise <4k, reward >8k
    if length_ratio < 0.5:
        # e.g. 0.25 -> -0.25, 0.0 -> -0.5
        length_bonus = -(0.5 - length_ratio)
    else:
        # 0.5 -> 0, 1.0+ -> 1.0 (clipped)
        length_bonus = min((length_ratio - 0.5) * 2.0, 1.0)


    # keep ppl gentle; clamp to avoid huge swings
    ppl_term = math.log1p(min(perplexity, 100.0))

    reward = (w_length * length_bonus) \
             + (w_prompt * prompt_validity) \
             - (w_eos * eos_penalty) \
             - 0.05 * ppl_term

    return float(reward), int(gen_length), float(prompt_validity), float(eos_penalty), float(perplexity)


def adjust_reward_weights(avg_gen_length,
                          avg_prompt_validity=None,
                          avg_eos_penalty=None):
    """
    Adapt reward weights online. When avg_prompt_validity is None, we skip
    adjusting w_prompt (length-only setup).
    """
    global w_length, w_prompt, w_eos

    # --- length adjustment (toward TARGET_GEN_LENGTH) ---
    try:
        if avg_gen_length is not None:
            delta_length = (TARGET_GEN_LENGTH - float(avg_gen_length)) / max(1.0, TARGET_GEN_LENGTH)
            w_length = max(0.01, min(1.0, w_length + 0.10 * delta_length))
    except Exception:
        # be robust to weird types
        pass

    # --- prompt validity adjustment (optional / skipped if None) ---
    try:
        if avg_prompt_validity is not None:
            delta_prompt = (TARGET_PROMPT_VALIDITY - float(avg_prompt_validity))
            # keep this small; or set to 0.0 permanently if you want it frozen
            w_prompt = max(0.0, min(0.30, w_prompt + 0.05 * delta_prompt))
        # else: leave w_prompt unchanged (length-only)
    except Exception:
        pass

    # --- eos penalty adjustment (optional) ---
    try:
        if avg_eos_penalty is not None:
            # eos_penalty is typically <= 0 for early EOS; increase w_eos when penalty is more negative
            w_eos = max(0.0, min(0.20, w_eos + 0.20 * (-float(avg_eos_penalty))))
    except Exception:
        pass

    return w_length, w_prompt, w_eos


def mutate_prompt(base_tokens, vocab_size, mutation_rate: float = 0.25,
                  top_k: int = 75) -> list[int]:
    """
    Mutate `base_tokens` in-place using the LM’s next-token distribution.
    •  never inserts ids 0-2
    •  keeps length in [3, 16]
    """
    mutated = base_tokens[:]

    # ---------- 1) token replacements ---------------------------------
    for i in range(len(mutated)):
        if random.random() >= mutation_rate:
            continue
        ctx = mutated[:i]                       # left context
        if not ctx or SIMULATE:                 # nothing to condition on
            continue

        _ensure_model()
        with torch.no_grad():
            logits = llama_model(
                torch.tensor(ctx, device=DEVICE).unsqueeze(0)
            ).logits[0, -1]

        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, k=top_k).indices.tolist()
        # skip <pad>=0, <unk>=1, <eos>=2
        candidates = [tid for tid in top_ids if tid > 2]
        if candidates:
            mutated[i] = random.choice(candidates)

    # ---------- 2) optional extension ---------------------------------
    if len(mutated) < 10 and random.random() < 0.5:
        if SIMULATE:
            next_tok = random.randint(3, vocab_size - 1)
        else:
            _ensure_model()
            with torch.no_grad():
                logits = llama_model(
                    torch.tensor(mutated, device=DEVICE).unsqueeze(0)
                ).logits[0, -1]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
        if next_tok <= 2:                        # final guard
            next_tok = 3
        mutated.append(next_tok)

    # ---------- 3) floor length ≥ 3 -----------------------------------
    if len(mutated) < 3:
        filler = tokenizer.encode("the")[0]      # harmless token
        mutated += [filler] * (3 - len(mutated))

    return mutated[:10]                          # hard cap



def generate_initial_prompt_from_llama(seed_text="In", max_tokens=20):
    _ensure_model()  
    input_ids = tokenizer.encode(seed_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = llama_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.strip()

logger.info(f"▶️  Starting reward-tuning for {NUM_EPISODES} episodes (seed={SEED})")

# ---------------------------
# TRAINING LOOP WITH ITERATIVE REWARD TUNING
# ---------------------------
def training_loop():
    global w_length, w_prompt, w_eos
    _ensure_model()  
    rewards_list = []
    gen_lengths_list = []
    prompt_validity_list = []

    # Load previously successful prompts (if any)
    try:
        with open("training_prompts.txt", "r") as f:
            successful_prompts = [
                line.strip() for line in f if line.strip() and not line.startswith("[")
            ]
    except FileNotFoundError:
        successful_prompts = []

    for episode in range(1, NUM_EPISODES + 1):
        if successful_prompts:
            base_prompt_text = random.choice(successful_prompts)
            base_tokens = tokenizer.encode(base_prompt_text, add_special_tokens=False)
            prompt_tokens = mutate_prompt(base_tokens, tokenizer.vocab_size)
            logger.debug(f"[DEBUG] Prompt tokens: {prompt_tokens}")
            logger.debug(f"[DEBUG] Prompt text: {tokenizer.decode(prompt_tokens, skip_special_tokens=True)}")
            # Occasionally pad with extra token to increase length
            if random.random() < 0.3 and len(prompt_tokens) < 16:
                prompt_tokens += [random.randint(3, tokenizer.vocab_size - 1)]
        else:
            llama_prompt_text = generate_initial_prompt_from_llama()
            base_tokens = tokenizer.encode(llama_prompt_text, add_special_tokens=False)
            prompt_tokens = mutate_prompt(base_tokens, tokenizer.vocab_size)
        curr_len_thresh = min(8192, 64 + episode * 48)
        curr_coh_thresh = min(0.75, max(0.15, 0.05 + episode * 0.004))
        if episode % 5 == 0 and len(gen_lengths_list) > 5:
            avg_gen_length = np.mean(gen_lengths_list[-5:])
            if avg_gen_length < curr_len_thresh * 0.5:
                curr_len_thresh = max(1024, curr_len_thresh - 512)

        logger.debug(f"Dynamic Thresholds - Length: {curr_len_thresh}, Coherence: {curr_coh_thresh}")

        reward, gen_length, prompt_validity, eos_penalty, perplexity = compute_reward(
            prompt_tokens, goal_len=curr_len_thresh
        )

        logger.info(f"[Episode {episode}] validity_score={prompt_validity:.3f}, curr_coh_thresh={curr_coh_thresh:.3f}")
        logger.debug(f"[DEBUG] Save check: gen_len={gen_length}, curr_len_thresh={curr_len_thresh}, valid={prompt_validity}, curr_coh_thresh={curr_coh_thresh}")
        logger.info(f"PPL={perplexity:.2f}")
        if gen_length >= 4096 and prompt_validity >= curr_coh_thresh:
            logger.info(f"[SAVE] val={prompt_validity:.3f}, len={gen_length}")
            prompt_str = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            metadata = {
                "prompt": prompt_str,
                "gen_len": gen_length,
                "reward": reward,
                "validity": prompt_validity,
                "episode": episode 
            }
            with open("training_prompts.jsonl", "a") as f:
                f.write(json.dumps(metadata) + "\n")
            with open("training_prompts.txt", "a") as f:
                f.write(prompt_str + "\n")

            # Log total prompts saved so far
            try:
                with open("training_prompts.txt", "r") as f:
                    successful_prompts = [line.strip() for line in f if line.strip()]
                logger.info(f"[SUMMARY] Saved {len(successful_prompts)} prompts so far.")
            except FileNotFoundError:
                logger.info("[SUMMARY] No prompts saved yet.")

        else:
            logger.warning(
                "[SKIP] len=%d valid=%.2f (need ≥%.2f and len 3-10)",
                len(prompt_tokens), prompt_validity, curr_coh_thresh
            )
            logger.warning(f"[SKIPPED] Prompt too short or low validity. Text: {tokenizer.decode(prompt_tokens)}")

        rewards_list.append(reward)
        gen_lengths_list.append(gen_length)
        prompt_validity_list.append(prompt_validity)
        recent_rewards.append(reward)
        recent_validities.append(prompt_validity)
        recent_eos_penalties.append(eos_penalty)
        logger.info(
            f"[Episode {episode:03d}/{NUM_EPISODES}]  "
            f"Reward={reward:.3f}  "
            f"GenLen={gen_length:<4d}  "
            f"Validity={prompt_validity:.3f}  "
            f"EOSpen={eos_penalty:.3f}"
        )

        # Adjust reward weights periodically
        if episode % ADJUST_INTERVAL == 0 and len(recent_rewards) == ROLL_WINDOW:
            avg_gen_length = np.mean(gen_lengths_list[-ROLL_WINDOW:])
            avg_prompt_validity = np.mean(recent_validities)
            avg_eos_penalty = np.mean(recent_eos_penalties)
            logger.info(f"Rolling Avg over last {ROLL_WINDOW} episodes: GenLength={avg_gen_length:.2f}, "
                        f"PromptValidity={avg_prompt_validity:.2f}, EOSPenalty={avg_eos_penalty:.2f}")
            adjust_reward_weights(avg_gen_length, avg_prompt_validity, avg_eos_penalty)

            logger.info(f"Avg over last {ADJUST_INTERVAL} episodes: GenLength={avg_gen_length:.2f}, "
                        f"PromptValidity={avg_prompt_validity:.2f}")


    # Save final weights
    with open(os.path.join("reward_weights.json"), "w") as f:
        json.dump({"w_length": w_length, "w_prompt": w_prompt, "w_eos": w_eos}, f)
    logger.info("Final reward weights saved.")
    # Log top 5 highest-reward prompts
    top_prompts = sorted(
        zip(rewards_list, gen_lengths_list, prompt_validity_list),
        key=lambda x: x[0],
        reverse=True
    )[:5]
    logger.info("Top 5 prompt reward stats (Reward, GenLen, Validity):")
    for r, g, v in top_prompts:
        logger.info(f"  Reward={r:.2f}, GenLen={g}, Validity={v:.2f}")
    logger.info("Finished reward-tuning.")
    logger.info(f"Final weights: w_length={w_length:.3f}, w_prompt={w_prompt:.3f}, w_eos={w_eos:.3f}")
    logger.info("Top 5 prompts by reward:")
    for r, g, v in sorted(zip(rewards_list, gen_lengths_list, prompt_validity_list),
                          key=lambda x: x[0], reverse=True)[:5]:
        logger.info(f"   Reward={r:.3f}, GenLen={g}, Validity={v:.3f}")

    return rewards_list, gen_lengths_list, prompt_validity_list

# ---------------------------
# MAIN
# ---------------------------
#if __name__ == "__main__":
#    rewards, gen_lengths, prompt_validities = training_loop()
#    avg_reward = np.mean(rewards)
#    avg_gen = np.mean(gen_lengths)
#    avg_prompt = np.mean(prompt_validities)
#    logger.info(f"Final Averages: Reward={avg_reward:.2f}, GenLength={avg_gen:.2f}, PromptValidity={avg_prompt:.2f}")
