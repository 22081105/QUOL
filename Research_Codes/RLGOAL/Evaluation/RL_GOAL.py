import os, time
import json
import math
import random
import logging
import numpy as np
import pickle
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
import llama_env
import gymnasium as gym
from gymnasium.envs.registration import register
import sys
import reward_tuning
reward_tuning.SIMULATE = False
import csv
from reward_tuning import set_max_new_gen, set_boot_cap, get_boot_cap
set_max_new_gen(16384)

# ─── Deterministic seeding for NumPy/CPU/PyTorch ───
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

MAX_STEPS = 64
#register(
#    id='LLaMAMultiStepEnv-v0',
#    entry_point='llama_env:LLaMAMultiStepEnv',
#    max_episode_steps=MAX_STEPS,  
#)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
LOG_PATH = "logs/training_rl.log"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("rl_goal_logger")   # 1 global logger for RL + reward
logger.setLevel(logging.DEBUG)                 # capture everything
logger.propagate = False                       # avoid duplicate root output

if not logger.handlers:                        # do this exactly once
    fh = logging.FileHandler(LOG_PATH, mode="a", delay=True)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)


def log_flush() -> None:
    """Flush all file/console handlers immediately."""
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()


# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./rl_goal/"
os.makedirs(SAVE_PATH, exist_ok=True)
TRACE_PATH = os.path.join(SAVE_PATH, "trace.txt")

# LLM-specific settings
BEST_GEN_LEN = 0  # global best generated length seen so far
EOS_TOKEN_REWARD_PENALTY = 5 
NUM_EPISODES = 500000   
BATCH_SIZE = 4              
PPO_EPOCHS = 6           
GAMMA = 0.995                
LAMBDA_GAE = 0.9          
EPS_CLIP = 0.2            
LR_POLICY = 2e-5
LR_VALUE = 2e-5
REPLAY_CAPACITY = 75000  
HER_FRACTION = 0.3 
ENTROPY_COEF = 0.005
VALUE_LOSS_COEF = 0.1
CLIP_RANGE_VALUE = 0.2
#COHERENCE_MIN = 0.4
#LENGTH_THRESHOLD = 10
START_GOAL_RANGE = (2048, 4096)    
END_GOAL_RANGE   = (8192, 16384) 
CHECKPOINT_INTERVAL = 250 

# ─── PPO DIAGNOSTICS ──────────────────────────────────────────────────────────
LOG_PPO_CSV = "logs/ppo_diagnostics.csv"
PPO_TRAIN_STEP = 0            # global counter for PPO mini-batch updates
LAST_100_REWARD_AVG = 0.0     # updated from train loop before each ppo_update call

def _ensure_diag_csv():
    os.makedirs(os.path.dirname(LOG_PPO_CSV), exist_ok=True)
    header = ["step","policy_loss","value_loss","entropy",
              "policy_grad_preclip","policy_grad_postclip",
              "value_grad_preclip","value_grad_postclip",
              "policy_update_L2","policy_update_rel",
              "value_update_L2","value_update_rel",
              "lr_policy","lr_value","avg_reward_last100"]
    if not os.path.exists(LOG_PPO_CSV):
        with open(LOG_PPO_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)

# Auto-ramp: start ~2k, add 2k every 5k episodes, cap at 16k. Monotonic.
def maybe_update_boot_cap(episode_idx: int):
    step_every = 5000
    base = 2048
    inc  = 2048
    proposed = min(16384, base + (episode_idx // step_every) * inc)
    if proposed > get_boot_cap():
        set_boot_cap(proposed)
        logger.info(f"[cap] Raised BOOT_CAP to {proposed}")

def _grad_norm_now(model: nn.Module) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            tot += float((g * g).sum().item())
    return math.sqrt(tot) if tot > 0.0 else 0.0

def _param_vec_norm(params) -> float:
    tot = 0.0
    for p in params:
        d = p.detach()
        tot += float((d * d).sum().item())
    return math.sqrt(tot) if tot > 0.0 else 0.0

def _update_norm(prev_params, model: nn.Module) -> tuple[float, float]:
    """Return (absolute L2 of delta, relative L2 = ||Δ|| / (||θ_prev|| + 1e-12))."""
    num = 0.0
    den = 0.0
    for p_prev, p_cur in zip(prev_params, model.parameters()):
        d = (p_cur.detach() - p_prev)
        num += float((d * d).sum().item())
        den += float((p_prev * p_prev).sum().item())
    abs_l2 = math.sqrt(num) if num > 0.0 else 0.0
    rel_l2 = abs_l2 / (math.sqrt(den) + 1e-12)
    return abs_l2, rel_l2

def _write_diag_row(row: dict):
    with open(LOG_PPO_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            row["step"], row["policy_loss"], row["value_loss"], row["entropy"],
            row["policy_grad_preclip"], row["policy_grad_postclip"],
            row["value_grad_preclip"], row["value_grad_postclip"],
            row["policy_update_L2"], row["policy_update_rel"],
            row["value_update_L2"], row["value_update_rel"],
            row["lr_policy"], row["lr_value"], row["avg_reward_last100"],
        ])
# ─────────────────────────────────────────────────────────────────────────────

# Allow importing RL_GOAL without pulling full LLaMA weights (used by infer_rl_goal.py).
SKIP_MAIN_MODEL_LOAD = os.getenv("RL_GOAL_SKIP_MAIN_MODEL_LOAD", "0") == "1"

# ---------------------------
# LOAD MAIN LLaMA MODEL
# ---------------------------
# Always load tokenizer/vocab (policy needs this), but optionally skip full LLaMA weights.
if not hasattr(sys.modules[__name__], "_models_loaded"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    EOS_TOKEN_ID = tokenizer.eos_token_id
    VOCAB_SIZE = tokenizer.vocab_size

    llama_model = None
    if not SKIP_MAIN_MODEL_LOAD:
        logger.info("Loading main LLaMA model…")
        llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        llama_model.gradient_checkpointing_enable()
        if DEVICE == "cpu":
            logger.warning("⚠️  LLaMA loaded on CPU -- expect very slow rollouts!")
        logger.info("Skipping coherence/discriminator model (length-only rewards).")

        # Only set reward_tuning models when we actually loaded weights
        reward_tuning.set_models(tokenizer, llama_model)

    setattr(sys.modules[__name__], "_models_loaded", True)



adjust_reward_weights = reward_tuning.adjust_reward_weights
#w_length  = reward_tuning.w_length
#w_prompt  = reward_tuning.w_prompt
#w_eos     = reward_tuning.w_eos
EOS_TOKEN_ID = tokenizer.eos_token_id
VOCAB_SIZE = tokenizer.vocab_size
logger.info(f"Main LM loaded on {DEVICE}, vocab size={VOCAB_SIZE}, EOS={EOS_TOKEN_ID}")

# --- English-ish token mask (ASCII letters / digits / common punct / space) ---
def _is_englishish(s: str) -> bool:
    if not s:
        return False
    try:
        s.encode("ascii")  # fast ASCII check
    except UnicodeEncodeError:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"-()[]{}<>/\\|@#$%^&*_+=~`")
    # Keep tokens that are whitespace or mostly allowed chars
    letters = sum(ch.isalpha() for ch in s)
    allowed_chars = sum((ch in allowed) for ch in s)
    return (allowed_chars >= max(1, int(0.8 * len(s)))) or (letters >= 1)

ENGLISH_TOKEN_MASK = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
for tid in range(VOCAB_SIZE):
    tok = tokenizer.decode([tid], skip_special_tokens=False)
    ENGLISH_TOKEN_MASK[tid] = _is_englishish(tok)
# always allow EOS
ENGLISH_TOKEN_MASK[EOS_TOKEN_ID] = True
ENGLISH_TOKEN_MASK = ENGLISH_TOKEN_MASK.to(DEVICE)

# ---------------------------
# ADAPTIVE HER FRACTION
# ---------------------------
def update_her_fraction(success_rate, current_fraction, target=0.4):
    error = target - success_rate
    adjustment = 0.15 * error
    new_fraction = min(1.0, max(0.2, current_fraction + adjustment))
    return new_fraction

    
# ---------------------------
# REPLAY BUFFER & HER UTILS
# ---------------------------
class Transition:
    """
    For multi-step:
      (obs, goal, action, log_prob, reward, done, next_obs, next_goal, value, achieved_goal)
    'action' is the single token chosen at each step.
    """
    __slots__ = [
        "obs", "goal", "action", "log_prob", "reward", 
        "done", "next_obs", "next_goal", "value", "achieved_goal"
    ]

    def __init__(self, obs, goal, action, log_prob, reward, done, next_obs, next_goal, value, achieved_goal):
        self.obs = obs
        self.goal = goal
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.next_goal = next_goal
        self.value = value
        self.achieved_goal = achieved_goal

class ReplayBuffer:
    """
    Prioritised Experience Replay with ring-buffer overwrite.
    """
    def __init__(self, capacity: int):
        self.capacity   = capacity
        self.buffer     = []
        self.priorities = []
        self.position   = 0          
    def add(self, transition, priority: float | None = None):
        """Append *or* overwrite one transition."""
        if priority is None:
            priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position]     = transition
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size: int, alpha: float = 0.6):
        size = len(self.buffer)
        if size == 0:
            return [], [] 
        batch_size = min(batch_size, size)
        priorities = np.asarray(self.priorities, dtype=np.float32)
        scaled     = priorities ** alpha
        probs      = scaled / np.maximum(scaled.sum(), 1e-8) 
        replace = batch_size > size
        indices = np.random.choice(size, batch_size, p=probs, replace = replace)
        samples = [self.buffer[i] for i in indices]
        return samples, indices
    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            idx = int(idx)                       # ensure plain Python int
            self.priorities[idx] = float(prio)
    def __len__(self):
        return len(self.buffer)

def her_relabel(transitions, her_fraction, max_goal=16384, epsilon=5):
    if not transitions:
        return transitions
    final_len = transitions[-1].achieved_goal
    new_goal  = min(final_len, max_goal)
    out = []
    for tr in transitions:
        if random.random() >= her_fraction:
            out.append(tr)
            continue
        distance = abs(tr.achieved_goal - new_goal)
        new_reward = 1.0 if distance <= epsilon else -distance / max(max_goal, 1e-8)
        #logger.debug(f"HER Relabeling - Original Goal: {tr.goal}, New Goal: {new_goal}, Distance: {distance}, Reward: {new_reward}")
        new_done   = tr.done or (distance <= epsilon)
        out.append(
            Transition(
                tr.obs, new_goal, tr.action, tr.log_prob,
                new_reward, new_done, tr.next_obs, new_goal,
                tr.value, tr.achieved_goal
            )
        )
    return out

# ---------------------------
# SOTA POLICY TRANSFORMER
# ---------------------------
class MiniPolicyTransformer(nn.Module):
    """
    Transformer-based policy that conditions on the partial token sequence
    plus a (bucketed) scalar goal. The policy crafts a short prompt; the victim
    does the long generation.
    """
    def __init__(self, hidden_dim: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # ---- embeddings ----
        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)

        # Positional table large enough for your observation length (0 is reserved for the goal slot).
        # Override via env var if needed, e.g., POLICY_MAX_POS=4096
        self.MAX_POS = int(os.getenv("POLICY_MAX_POS", "2048"))
        self.pos_embedding = nn.Embedding(self.MAX_POS, hidden_dim)

        # Bucket the raw goal (e.g., 0..16384) into 65 bins (0..64 inclusive).
        # By default, with GOAL_BUCKET_SIZE=256, 16384//256 == 64 → fits.
        self.GOAL_BUCKETS     = 65
        self.GOAL_BUCKET_SIZE = int(os.getenv("GOAL_BUCKET_SIZE", "256"))
        self.goal_embedding   = nn.Embedding(self.GOAL_BUCKETS, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out     = nn.Linear(hidden_dim, VOCAB_SIZE)

        # Precompute rough char-length per token for a mild length bias
        tok_lens = torch.tensor(
            [len(tokenizer.decode([tid])) for tid in range(VOCAB_SIZE)],
            dtype=torch.float32
        )
        self.register_buffer("tok_lengths", tok_lens)   # moves with .to(device)
        self.length_bias = 0.4

    def _bucket_goal(self, goal, device):
        """
        Map raw integer goal (e.g., 256..16384) to [0..64] using GOAL_BUCKET_SIZE.
        Accepts int or LongTensor; returns LongTensor on 'device'.
        """
        if not isinstance(goal, torch.Tensor):
            g = torch.tensor(goal, device=device, dtype=torch.long)
        else:
            g = goal.to(device=device, dtype=torch.long)
        g_idx = torch.clamp(g // self.GOAL_BUCKET_SIZE, 0, self.GOAL_BUCKETS - 1)
        return g_idx

    def forward(self,
                tokens: torch.Tensor,   # (seq_len,) long
                goal:   int,
                return_hidden: bool = True):
        """
        • tokens are already on the correct device
        • if return_hidden=True, also returns per-layer hidden states
        """

        device = tokens.device

        # ---- empty sequence special-case ----
        if tokens.numel() == 0:
            g_idx    = self._bucket_goal(goal, device)
            goal_vec = self.goal_embedding(g_idx.view(1)) \
                       + self.pos_embedding(torch.tensor([0], device=device))
            x = goal_vec.unsqueeze(1)  # (1, 1, hidden)
            hidden_states = []
            for layer in self.transformer.layers:
                x = layer(x)
                hidden_states.append(x.squeeze(1))
            logits = self.fc_out(self.layer_norm(x[0, 0]))
            logits = logits + self.length_bias * self.tok_lengths
            return logits if not return_hidden else (logits, hidden_states)

        # ---- token + position embeddings ----
        seq_len = tokens.size(0)
        if seq_len + 1 > self.pos_embedding.num_embeddings:
            raise RuntimeError(
                f"POLICY_MAX_POS too small for seq_len={seq_len}+goal_slot; "
                f"have {self.pos_embedding.num_embeddings}. Set env POLICY_MAX_POS higher."
            )

        tok_emb = self.token_embedding(tokens)  # (L, H)

        # positions: 0 is the goal slot, tokens occupy 1..L
        pos_idx = torch.arange(1, seq_len + 1, device=device)
        pos_emb = self.pos_embedding(pos_idx)   # (L, H)

        x = self.dropout(tok_emb + pos_emb)     # (L, H)

        # ---- prepend goal "token" at position 0 ----
        g_idx    = self._bucket_goal(goal, device)
        goal_vec = self.goal_embedding(g_idx.view(1)) \
                   + self.pos_embedding(torch.tensor([0], device=device))  # (1, H)
        x = torch.cat([goal_vec, x], dim=0).unsqueeze(1)  # (L+1, 1, H)

        # ---- transformer stack ----
        hidden_states = []
        for layer in self.transformer.layers:
            x = layer(x)
            hidden_states.append(x.squeeze(1))

        # ---- project goal position (index 0) ----
        goal_out = self.layer_norm(x[0, 0])    # (H,)
        logits   = self.fc_out(goal_out)       # (V,)
        logits   = logits + self.length_bias * self.tok_lengths
        return (logits, hidden_states) if return_hidden else logits



# ---------------------------
# SOTA VALUE TRANSFORMER
# ---------------------------
class MiniValueTransformer(nn.Module):
    """
    Transformer-based value network: returns a scalar V(s, g).
    Uses the same goal bucketing and positional scheme as the policy.
    """
    def __init__(self, hidden_dim: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)

        self.MAX_POS = int(os.getenv("POLICY_MAX_POS", "2048"))
        self.pos_embedding = nn.Embedding(self.MAX_POS, hidden_dim)

        self.GOAL_BUCKETS     = 65
        self.GOAL_BUCKET_SIZE = int(os.getenv("GOAL_BUCKET_SIZE", "256"))
        self.goal_embedding   = nn.Embedding(self.GOAL_BUCKETS, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out     = nn.Linear(hidden_dim, 1)

    def _bucket_goal(self, goal, device):
        if not isinstance(goal, torch.Tensor):
            g = torch.tensor(goal, device=device, dtype=torch.long)
        else:
            g = goal.to(device=device, dtype=torch.long)
        g_idx = torch.clamp(g // self.GOAL_BUCKET_SIZE, 0, self.GOAL_BUCKETS - 1)
        return g_idx

    def forward(self,
                tokens: torch.Tensor,  # (seq_len,) long
                goal:   int,
                return_hidden: bool = True):
        device = tokens.device

        # empty sequence
        if tokens.numel() == 0:
            g_idx    = self._bucket_goal(goal, device)
            goal_vec = self.goal_embedding(g_idx.view(1)) \
                       + self.pos_embedding(torch.tensor([0], device=device))
            x = goal_vec.unsqueeze(1)                # (1, 1, H)
            hidden_states = []
            for layer in self.transformer.layers:
                x = layer(x)
                hidden_states.append(x.squeeze(1))
            value = self.fc_out(self.layer_norm(x[0, 0]))  # (1,)
            return value if not return_hidden else (value, hidden_states)

        seq_len = tokens.size(0)
        if seq_len + 1 > self.pos_embedding.num_embeddings:
            raise RuntimeError(
                f"POLICY_MAX_POS too small for seq_len={seq_len}+goal_slot; "
                f"have {self.pos_embedding.num_embeddings}. Set env POLICY_MAX_POS higher."
            )

        tok_emb = self.token_embedding(tokens)  # (L, H)
        pos_idx = torch.arange(1, seq_len + 1, device=device)
        pos_emb = self.pos_embedding(pos_idx)   # (L, H)
        x = self.dropout(tok_emb + pos_emb)

        g_idx    = self._bucket_goal(goal, device)
        goal_vec = self.goal_embedding(g_idx.view(1)) \
                   + self.pos_embedding(torch.tensor([0], device=device))
        x = torch.cat([goal_vec, x], dim=0).unsqueeze(1)   # (L+1, 1, H)

        hidden_states = []
        for layer in self.transformer.layers:
            x = layer(x)
            hidden_states.append(x.squeeze(1))

        value = self.fc_out(self.layer_norm(x[0, 0]))      # (1,)
        return (value, hidden_states) if return_hidden else value


def log_training_metrics(policy_loss, value_loss, entropy_list, optimizer_policy, optimizer_value, step):
    avg_entropy = sum(entropy_list) / len(entropy_list) if entropy_list else 0.0
    policy_lr = optimizer_policy.param_groups[0]['lr']
    value_lr = optimizer_value.param_groups[0]['lr']
    logger.info(f"[Step {step}] PolicyLoss={policy_loss:.4f}, ValueLoss={value_loss:.4f}, Entropy={avg_entropy:.4f}, LR_Policy={policy_lr:.6f}, LR_Value={value_lr:.6f}")

# ---------------------------
# PPO + GAE
# ---------------------------
def compute_gae(transitions, gamma=0.99, lam=0.95):
    """
    transitions: list of Transition objects for one *entire episode*.
    We'll compute GAE-lambda for advantage estimation.
    Return a list of (transition, normalized_advantage, normalized_return).
    """
    advantages = []
    returns = []
    gae = 0
    for i in reversed(range(len(transitions))):
        if i == len(transitions) - 1:
            next_value = 0  # no next state if done
            next_done = 1.0
        else:
            next_value = transitions[i+1].value
            next_done = 0.0 if not transitions[i+1].done else 1.0

        delta = transitions[i].reward + gamma * next_value * (1 - next_done) - transitions[i].value
        gae = delta + gamma * lam * (1 - next_done) * gae
        adv = gae
        advantages.insert(0, adv)
        returns.insert(0, adv + transitions[i].value)

    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    ret_tensor = torch.tensor(returns, dtype=torch.float32)

    # Normalize both advantages and returns
    norm_adv = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    norm_ret = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)

    # Optional logging
    # print(f"[GAE DEBUG] Adv: mean={adv_tensor.mean():.3f}, std={adv_tensor.std():.3f}")
    # print(f"[GAE DEBUG] Ret: mean={ret_tensor.mean():.3f}, std={ret_tensor.std():.3f}")

    out = []
    for t, adv, ret in zip(transitions, norm_adv.tolist(), norm_ret.tolist()):
        out.append((t, adv, ret))
    return out


def ppo_update(policy_net, value_net,
               optimizer_policy, optimizer_value,
               batch,
               scheduler_policy=None, scheduler_value=None,
               eps_clip=0.2, epochs=4):
    MAX_CTX = 1024
    for _ in range(epochs):
        random.shuffle(batch)
        for sub_i in range(0, len(batch), BATCH_SIZE):
            mini_batch = batch[sub_i:sub_i+BATCH_SIZE]           
            policy_loss = 0.0
            value_loss = 0.0
            entropy_list = []

            for (transition, adv, ret) in mini_batch:
                tokens_list = transition.obs
                if len(tokens_list) > MAX_CTX:
                    tokens_list = tokens_list[-MAX_CTX:]
                tokens = torch.tensor(tokens_list, dtype=torch.long, device=DEVICE)
                goal = transition.goal
                action = transition.action
                old_log_prob = transition.log_prob  

                # ==== Policy update ====
                logits = policy_net(tokens, goal, return_hidden=False)
                if not torch.isfinite(logits).all():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(torch.tensor(action, device=DEVICE))
                delta_logp = (new_log_prob - old_log_prob).clamp(-10, 10)
                ratio = torch.exp(delta_logp)

                adv_t = torch.tensor(adv, dtype=torch.float, device=DEVICE)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_t
                loss_pi = -torch.min(surr1, surr2)
                entropy = dist.entropy()
                policy_loss += loss_pi - ENTROPY_COEF * entropy
                entropy_list.append(entropy.item())

                # ==== Value update with clipping and return normalization ====
                new_value = value_net(tokens, goal, return_hidden=False).squeeze()
                ret_t = torch.tensor(ret, dtype=torch.float, device=DEVICE)

                # Use clipped value loss (PPO2-style) – compare against rollout-time value
                old_value = torch.tensor(transition.value, dtype=torch.float32, device=DEVICE)
                value_clipped = old_value + (new_value - old_value).clamp(-CLIP_RANGE_VALUE, CLIP_RANGE_VALUE)
                loss_unclipped = (new_value - ret_t) ** 2
                loss_clipped  = (value_clipped - ret_t) ** 2
                loss_vf = torch.max(loss_unclipped, loss_clipped)
                value_loss += loss_vf


            # Average losses
            policy_loss /= len(mini_batch)
            value_loss = VALUE_LOSS_COEF * (value_loss / len(mini_batch))

            # Log metrics
            log_training_metrics(policy_loss.detach().item(), value_loss.detach().item(),
                                 entropy_list, optimizer_policy, optimizer_value, step=sub_i)

            # Backward + optimizer with diagnostics
            global PPO_TRAIN_STEP, LAST_100_REWARD_AVG
            
            # Snapshot params before update (to measure Δθ)
            prev_policy_params = [p.detach().clone() for p in policy_net.parameters()]
            prev_value_params  = [v.detach().clone() for v in value_net.parameters()]
            
            # --- POLICY ---
            optimizer_policy.zero_grad()
            policy_loss.backward()
            policy_grad_pre = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # returns pre-clip norm
            policy_grad_post = _grad_norm_now(policy_net)
            optimizer_policy.step()
            pol_abs_upd, pol_rel_upd = _update_norm(prev_policy_params, policy_net)
            
            # --- VALUE ---
            optimizer_value.zero_grad()
            value_loss.backward()
            value_grad_pre = torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            value_grad_post = _grad_norm_now(value_net)
            optimizer_value.step()
            val_abs_upd, val_rel_upd = _update_norm(prev_value_params, value_net)
            
            # Write one diagnostics row per mini-batch update
            entropy_mean = float(sum(entropy_list) / len(entropy_list)) if entropy_list else 0.0
            lr_pol = (scheduler_policy.get_last_lr()[0]
                      if scheduler_policy is not None
                      else optimizer_policy.param_groups[0]["lr"])
            lr_val = (scheduler_value.get_last_lr()[0]
                      if scheduler_value is not None
                      else optimizer_value.param_groups[0]["lr"])
            
            _write_diag_row({
                "step": PPO_TRAIN_STEP,
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss":  float(value_loss.detach().item()),
                "entropy":     entropy_mean,
                "policy_grad_preclip":  float(policy_grad_pre),
                "policy_grad_postclip": float(policy_grad_post),
                "value_grad_preclip":   float(value_grad_pre),
                "value_grad_postclip":  float(value_grad_post),
                "policy_update_L2": float(pol_abs_upd),
                "policy_update_rel": float(pol_rel_upd),
                "value_update_L2":  float(val_abs_upd),
                "value_update_rel": float(val_rel_upd),
                "lr_policy": float(lr_pol),
                "lr_value":  float(lr_val),
                "avg_reward_last100": float(LAST_100_REWARD_AVG),
            })
            PPO_TRAIN_STEP += 1



# ---------------------------
# CHECKPOINTING
# ---------------------------

def save_checkpoint(episode,
                    best_reward,
                    policy_net,
                    value_net,
                    optimizer_policy,
                    optimizer_value,
                    scheduler_policy=None,
                    scheduler_value=None,
                    save_path="./rl_goal/",
                    is_best: bool = False,
                    do_len_save: bool = False,
                    len_save_tag: str | None = None):
    import os, torch, random, numpy as np

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"checkpoint_ep{episode}.pt")

    # RNG state
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()

    ckpt = {
        "episode": episode,
        "best_reward": best_reward,
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "optimizer_policy_state": optimizer_policy.state_dict(),
        "optimizer_value_state": optimizer_value.state_dict(),
        "rng_state": rng_state,
    }

    # schedulers are optional and may not exist in old runs
    if scheduler_policy is not None:
        ckpt["scheduler_policy_state"] = scheduler_policy.state_dict()
    if scheduler_value is not None:
        ckpt["scheduler_value_state"]  = scheduler_value.state_dict()

    torch.save(ckpt, filename)
    logger.info(f"Checkpoint saved: {filename}")

    if is_best:
        best_path = os.path.join(save_path, "best.pt")
        torch.save(ckpt, best_path)
        logger.info(f"Best model updated at {best_path}")

    if do_len_save:
        tag = len_save_tag or "len16000plus"
        special_path = os.path.join(save_path, f"{tag}_ep{episode}.pt")
        torch.save(ckpt, special_path)
        logger.info(f"[LEN-SAVE] Saved 16k+ checkpoint → {special_path}")
        
def load_checkpoint(policy_net,
                    value_net,
                    optimizer_policy,
                    optimizer_value,
                    scheduler_policy=None,
                    scheduler_value=None,
                    save_path="./rl_goal/"):
    import os, re, glob, torch, random, numpy as np

    ckpts = glob.glob(os.path.join(save_path, "checkpoint_ep*.pt"))
    if not ckpts:
        logger.info("No checkpoints found. Starting from scratch.")
        return 0, float("-inf")

    def _ep_from_name(fp):
        m = re.search(r"checkpoint_ep(\d+)\.pt$", os.path.basename(fp))
        return int(m.group(1)) if m else -1

    path = max(ckpts, key=_ep_from_name)
    logger.info("[ckpt] loading %s", os.path.basename(path))
    data = torch.load(path, map_location="cpu")

    def _maybe_strip_module(sd: dict):
        if not sd:
            return sd
        ks = list(sd.keys())
        if ks and all(k.startswith("module.") for k in ks):
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    def _filtered_load(model, src_sd, name):
        src_sd = _maybe_strip_module(src_sd or {})
        tgt_sd = model.state_dict()
        new_sd, skipped = {}, []
        for k, v in src_sd.items():
            if k not in tgt_sd:
                skipped.append((k, "missing_in_model"))
                continue
            if tuple(v.shape) != tuple(tgt_sd[k].shape):
                skipped.append((k, f"{tuple(v.shape)} -> {tuple(tgt_sd[k].shape)}"))
                continue
            new_sd[k] = v
        # load what matches; leave others at init
        model.load_state_dict(new_sd, strict=False)
        if skipped:
            logger.warning("[ckpt:%s] skipped %d tensors (shape/key mismatch), e.g. %s",
                           name, len(skipped), skipped[:3])
        return bool(skipped)

    # ----- models -----
    p_sd = data.get("policy_state_dict") or data.get("policy")
    v_sd = data.get("value_state_dict")  or data.get("value")
    p_skipped = _filtered_load(policy_net, p_sd, "policy")
    v_skipped = _filtered_load(value_net,  v_sd, "value")
    arch_changed = (p_skipped or v_skipped)

    # ----- episode / reward -----
    start_episode = int(data.get("episode", 0))
    best_reward   = float(data.get("best_reward", float("-inf")))

    # ----- optimizers / schedulers (only if no skips) -----
    if not arch_changed:
        try:
            pol_opt_sd = (data.get("optimizer_policy_state_dict")
                          or data.get("optimizer_policy_state")
                          or data.get("optimizer_policy"))
            val_opt_sd = (data.get("optimizer_value_state_dict")
                          or data.get("optimizer_value_state")
                          or data.get("optimizer_value"))
            if pol_opt_sd and val_opt_sd:
                optimizer_policy.load_state_dict(pol_opt_sd)
                optimizer_value.load_state_dict(val_opt_sd)
            if scheduler_policy is not None:
                pol_sch_sd = (data.get("scheduler_policy_state_dict")
                              or data.get("scheduler_policy_state"))
                if pol_sch_sd:
                    scheduler_policy.load_state_dict(pol_sch_sd)
            if scheduler_value is not None:
                val_sch_sd = (data.get("scheduler_value_state_dict")
                              or data.get("scheduler_value_state"))
                if val_sch_sd:
                    scheduler_value.load_state_dict(val_sch_sd)
            logger.info("[ckpt] optimizers/schedulers loaded")
        except Exception as e:
            logger.warning("[ckpt] optimizer/scheduler load failed; reinit. err=%s", e)
    else:
        logger.warning("[ckpt] architecture changed; not loading optimizer/scheduler (reinit).")

    # ----- RNG restore (best-effort) -----
    rng = data.get("rng_state") or {}
    try:
        if "python" in rng: random.setstate(rng["python"])
        if "numpy"  in rng: np.random.set_state(rng["numpy"])
        if "torch"  in rng: torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"])
    except Exception as e:
        logger.warning("[ckpt] RNG restore warning: %s", e)

    logger.info("[ckpt] resumed at episode %d (best_reward=%s)", start_episode, best_reward)
    return start_episode, best_reward

# ------------------------------------------------------------------
def safe_tokens(seq: list[int]) -> list[int]:
    """
    Remove all ids ≤ 2 and cap length to 16 tokens.
    Guarantees at least 3 non-special ids (fallback 3,4,5).
    """
    cleaned = [t for t in seq if t > 2]
    return cleaned[:10] if len(cleaned) >= 3 else [3, 4, 5]
# ------------------------------------------------------------------
def run_episode(episode):
    """
    Lightweight one-off run (random actions). Keeps env plumbing healthy.
    Returns: (total_reward, gen_length, validity_score, eos_penalty)
    """
    env = gym.make("LLaMAMultiStepEnv-v0")
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
    gen_length   = info.get("gen_length", 0)
    validity     = float(info.get("prompt_validity", 0.0))
    eos_penalty  = float(info.get("eos_penalty", 0.0))
    return total_reward, gen_length, validity, eos_penalty

def maybe_update_max_new_gen(episode_idx: int):
    steps = episode_idx // 200
    proposed = min(16384, 256 + steps * 256)
    if proposed > reward_tuning.MAX_NEW_GEN:
        reward_tuning.set_max_new_gen(proposed)
        
# Pool file paths (reader already looks for these names)
POOL_TXT   = os.environ.get("PROMPT_POOL_TXT",   "training_prompts.txt")
POOL_JSONL = os.environ.get("PROMPT_POOL_JSONL", "training_prompts.jsonl")

def _append_prompt_record(tokenizer, tokens, gen_len, goal, episode_idx, env_idx):
    """Save final composed prompt (seed + policy tokens) to TXT and JSONL."""
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception:
        text = " ".join(str(int(t)) for t in tokens)
    flat = text.replace("\n", " ").strip()

    rec = {
        "ts": time.time(),
        "episode": int(episode_idx),
        "env": int(env_idx),
        "goal": int(goal),
        "gen_len": int(gen_len),
        "prompt": flat,
        "tokens": [int(t) for t in tokens],
    }

    # TXT
    with open(POOL_TXT, "a", encoding="utf-8") as f:
        f.write(flat + "\n")
        f.flush()
        os.fsync(f.fileno())
    
    # JSONL
    with open(POOL_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def train_sota_multistep():
    global HER_FRACTION, LAST_100_REWARD_AVG, BEST_GEN_LEN
    best_gen_len = BEST_GEN_LEN
    start_episode = 0
    HER_FRACTION = 0.3
    logger.info(f"=== STARTING RL-GOAL TRAIN for {NUM_EPISODES} episodes ===")
    log_flush()
    _ensure_diag_csv()
    def make_env():
        return gym.make("LLaMAMultiStepEnv-v0")
    batched_env = SyncVectorEnv([make_env for _ in range(BATCH_SIZE)])
    policy_net = MiniPolicyTransformer(hidden_dim=256, n_layers=2, n_heads=4).to(DEVICE)
    value_net  = MiniValueTransformer (hidden_dim=256, n_layers=2, n_heads=4).to(DEVICE)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LR_POLICY)
    optimizer_value  = optim.Adam(value_net.parameters(),  lr=LR_VALUE)
    scheduler_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy, step_size=500, gamma=0.9)
    scheduler_value  = torch.optim.lr_scheduler.StepLR(optimizer_value,  step_size=500, gamma=0.9)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    start_episode, best_reward = load_checkpoint(policy_net, value_net, optimizer_policy, optimizer_value, scheduler_policy, scheduler_value, SAVE_PATH)
    PROMPT_PHASE_OUT_EPISODE = int(NUM_EPISODES * 0.2)
    all_rewards          = []
    episodes_history     = []
    gen_lengths_list     = []
    eos_penalties_list   = []
    total_episodes = NUM_EPISODES - start_episode
    num_batches = math.ceil(total_episodes / BATCH_SIZE)
    last_ckpt_episode = (start_episode // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
    seen_prompts: set[str] = set()
    success_history = []

    def _load_prompt_pool(tokenizer, limit=None):
        """Load prompts from training_prompts.jsonl (preferred) or training_prompts.txt.
        Returns a list of (text, tokens, score) where score≈past gen_len if available."""
        pool = []
    
        # Prefer JSONL (has gen_len)
        try:
            if os.path.isfile("training_prompts.jsonl") and os.path.getsize("training_prompts.jsonl") > 0:
                with open("training_prompts.jsonl", "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj  = json.loads(ln)
                            text = (obj.get("prompt") or "").strip()
                            if not text:
                                continue
                            toks = tokenizer.encode(text, add_special_tokens=False)
                            if len(toks) >= 3:
                                pool.append((text, toks, int(obj.get("gen_len", 0))))
                        except Exception:
                            continue
        except FileNotFoundError:
            pass
    
        # Fallback to TXT if JSONL empty
        if not pool:
            try:
                if os.path.isfile("training_prompts.txt") and os.path.getsize("training_prompts.txt") > 0:
                    with open("training_prompts.txt", "r", encoding="utf-8") as f:
                        for ln in f:
                            text = ln.strip()
                            if not text:
                                continue
                            toks = tokenizer.encode(text, add_special_tokens=False)
                            if len(toks) >= 3:
                                pool.append((text, toks, 0))
            except FileNotFoundError:
                pass
    
        if limit and len(pool) > limit:
            pool = pool[-limit:]  # keep freshest
        return pool

    def _choose_pool_prompt():
        # Try text file pool first
        try:
            with open("training_prompts.txt", "r") as f:
                prompt_pool = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            prompt_pool = []
    
        if prompt_pool:
            return random.choice(prompt_pool)
    
        # Fallback: jsonl pool with metadata
        try:
            with open("training_prompts.jsonl", "r") as f:
                prompt_data = []
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    text  = entry.get("prompt", "")
                    if text:
                        prompt_data.append(text)
        except FileNotFoundError:
            prompt_data = []
    
        if prompt_data:
            return random.choice(prompt_data)
    
        # Final fallback: base prompt
        return "The following text is generated:"
    
    for batch_idx in range(num_batches):
        episode_idx = start_episode + batch_idx * BATCH_SIZE
        maybe_update_max_new_gen(episode_idx)
        maybe_update_boot_cap(episode_idx)  
        if episode_idx >= NUM_EPISODES:
            break
    
        # --- Decayed mix between base and pool (broadcast one seed to all sub-envs) ---
        pool = _load_prompt_pool(tokenizer, limit=5000)
        
        # Decay from p_base≈1.0 -> 0.10 over PROMPT_PHASE_OUT_EPISODE
        p_base = 1.0 - min(1.0, episode_idx / max(1, PROMPT_PHASE_OUT_EPISODE))
        p_base = max(0.10, p_base)  # keep at least 10% base for exploration
        
        if pool and (random.random() > p_base):
            # Weighted by prior gen_len if present (pool entries: (text, toks, score))
            weights = [max(1, rec[2]) for rec in pool]
            tot = float(sum(weights))
            probs = [w / tot for w in weights]
            idx = np.random.choice(len(pool), p=probs)
            prompt_text = pool[idx][0]  # text
        else:
            prompt_text = "The following text is generated:"
        
        pool_tokens = [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=False) if int(t) > 2]

        boot_cap = get_boot_cap()

        # Upper bound: curriculum ramp, MAX_NEW_GEN, and current boot cap
        curr_cap = START_GOAL_RANGE[1] + (episode_idx // 10) * 50
        high = min(reward_tuning.MAX_NEW_GEN, curr_cap, boot_cap)

        # Lower bound: start in the multi-k regime
        low = START_GOAL_RANGE[0]
        if high < low:
            high = low

        # Bias 50% of the time toward the upper half [mid, high]
        mid = (low + high) // 2
        bias_high = 0.8 if boot_cap >= 8192 else 0.5  # after 8k, focus on upper half
        
        if random.random() < bias_high:
            goal_sample = random.randint(mid, high)
        else:
            goal_sample = random.randint(low, high)

        batch_goal = int(goal_sample)

        
        # Reset envs with both the prompt and the goal (SyncVectorEnv broadcasts options)
        obs, reset_infos = batched_env.reset(options={
            "base_prompt_tokens": pool_tokens,
            "goal": batch_goal
        })
        
        logger.info(f"[Batch {batch_idx}] Starting ep {episode_idx}, goal={batch_goal} (boot_cap={boot_cap}, max_new_gen={reward_tuning.MAX_NEW_GEN})")
        logger.debug(f"[TRAIN LOOP] Reset infos: {reset_infos}")
        
        # Mirror env goal locally for bookkeeping
        batch_goals = [batch_goal for _ in range(BATCH_SIZE)]
        
        log_flush()


        # 5) Build per-env token buffers directly from env observations (drop pad/specials <=2)
        current_tokens   = [[int(t) for t in obs_i.tolist() if int(t) > 2] for obs_i in obs]
        prompt_len       = [len(tok) for tok in current_tokens]
        original_prompts = [tok[:] for tok in current_tokens]

        dones             = [False] * BATCH_SIZE
        transitions_batch = [[]      for _ in range(BATCH_SIZE)]
        step_count        = 0
        while not all(dones) and step_count < MAX_STEPS:
            batch_actions   = []
            batch_log_probs = []
            batch_values    = []
            for i, (done, tokens, goal) in enumerate(zip(dones, current_tokens, batch_goals)):
                if not done:
                    ctx = tokens[-1024:]
                    ctx_tensor = (
                        torch.tensor(ctx, dtype=torch.long, device=DEVICE)
                        if ctx else torch.empty(0, dtype=torch.long, device=DEVICE)
                    )
                    # Diagnostic Logging Before Logits Calculation
                    #logger.debug(f"Current prompt tokens (last 10): {tokens[-10:]}")
                    #logger.debug(f"Goal: {goal}, Generated Length: {len(tokens)}")
                    #logger.debug(f"Context Window Length: {len(ctx)}")
                    
                    with torch.no_grad():
                        logits = policy_net(ctx_tensor, goal, return_hidden=False)
                    
                    # Apply temperature ...
                    TEMPERATURE = 0.8
                    logits = logits / max(TEMPERATURE, 1e-6)
                    
                    # Disallow EOS and non-English-ish tokens for the policy
                    logits[EOS_TOKEN_ID] = float("-inf")
                    logits[~ENGLISH_TOKEN_MASK] = float("-inf")
                    
                    # --- ε-mixture exploration: q = (1-ε)*softmax(logits) + ε*Uniform(valid) ---
                    
                    # 1) Valid set = tokens that are not masked to -inf
                    valid_mask = torch.isfinite(logits) & (logits > -1e30)
                    num_valid  = int(valid_mask.sum().item())
                    if num_valid == 0:
                        raise RuntimeError("No valid actions left after masking.")
                    
                    uniform_probs = torch.zeros_like(logits, dtype=torch.float)
                    uniform_probs[valid_mask] = 1.0 / num_valid
                    
                    # 2) Policy softmax
                    pi_probs = torch.softmax(logits, dim=-1)
                    
                    # 3) Epsilon schedule (decay over episodes)
                    EPS_START, EPS_END, EPS_DECAY_EPISODES = 0.01, 0.001, 20_000
                    eps_frac = max(0.0, min(1.0, 1.0 - (episode_idx / float(EPS_DECAY_EPISODES))))
                    epsilon  = EPS_END + (EPS_START - EPS_END) * eps_frac
                    
                    # 4) Mixture distribution
                    q = (1.0 - epsilon) * pi_probs + epsilon * uniform_probs
                    
                    # 5) Sample from q and store log q(a) as behavior log-prob
                    action_dist = torch.distributions.Categorical(probs=q)
                    action = int(action_dist.sample().item())
                    logp  = action_dist.log_prob(torch.tensor(action, device=DEVICE))


                    
                    with torch.no_grad():
                        val = (
                            value_net(ctx_tensor, goal, return_hidden=False).item()
                            if ctx_tensor.numel() > 0 else 0.0
                        )
                else:
                    action, logp, val = 0, torch.tensor(0.0, device=DEVICE), 0.0
            
                batch_actions.append(action)
                batch_log_probs.append(logp)
                batch_values.append(val)
            
            next_obs, rewards, terminations, truncations, step_infos = batched_env.step(batch_actions)
            
            # Logging the raw reward values received from the environment
            #logger.debug(f"Rewards received from env step: {rewards}")
            done_flags = np.logical_or(terminations, truncations).tolist()
            validities_raw = []
            if isinstance(step_infos, list):
                for i in range(BATCH_SIZE):
                    #logger.debug(f"[DEBUG TRACE] Step info for env {i}: {step_infos[i]}")
                    info = step_infos[i] if isinstance(step_infos[i], dict) else {}
                    validity = info.get("prompt_validity", 0.0)
                    validities_raw.append(validity)
            elif isinstance(step_infos, dict):  # fallback case
                for i in range(BATCH_SIZE):
                    env_info = step_infos.get(str(i), {}) or step_infos.get(i, {})
                    #logger.debug(f"[DEBUG TRACE] Step info for env {i}: {env_info}")
                    validity = env_info.get("prompt_validity", 0.0)
                    validities_raw.append(validity)
            
            if isinstance(step_infos, dict):
                def _arr_or_default(key, fill=0, dtype=float):
                    arr = step_infos.get(key, None)
                    if arr is None:  # missing mid-episode
                        return np.full((BATCH_SIZE,), fill, dtype=dtype).tolist()
                    return arr.tolist()
                gen_lens        = _arr_or_default("gen_length",     fill=-1, dtype=int)
                eos_penalties   = _arr_or_default("eos_penalty",    fill=0.0, dtype=float)
                progress_ratios = _arr_or_default("progress_ratio", fill=0.0, dtype=float)
                perplexities    = _arr_or_default("perplexity",     fill=0.0, dtype=float)
                validities      = [1.0] * BATCH_SIZE
            else:
                gen_lens        = [d.get("gen_length",     -1) for d in step_infos]
                eos_penalties   = [d.get("eos_penalty",     0) for d in step_infos]
                progress_ratios = [d.get("progress_ratio",  0) for d in step_infos]
                perplexities    = [d.get("perplexity",      0) for d in step_infos]
                validities      = [1.0] * len(gen_lens)

            # --- merge terminal stats for envs that finished on this step ---
            # Gymnasium vector envs: dict form exposes an array at step_infos["final_info"]
            # Some builds/list forms: terminal stats may be in step_infos[i] directly or under .get("final_info")
            if isinstance(step_infos, dict) and "final_info" in step_infos and step_infos["final_info"] is not None:
                final_infos = step_infos["final_info"]
                for i, fi in enumerate(final_infos):
                    if i < len(done_flags) and done_flags[i] and fi is not None:
                        gen_lens[i]        = int(fi.get("gen_length",       gen_lens[i]))
                        eos_penalties[i]   = float(fi.get("eos_penalty",    eos_penalties[i]))
                        progress_ratios[i] = float(fi.get("progress_ratio", progress_ratios[i]))
                        perplexities[i]    = float(fi.get("perplexity",     perplexities[i]))
            elif isinstance(step_infos, list):
                for i in range(min(len(step_infos), len(done_flags))):
                    if done_flags[i] and isinstance(step_infos[i], dict):
                        fi = step_infos[i].get("final_info", step_infos[i])  # fall back to the dict itself
                        gen_lens[i]        = int(fi.get("gen_length",       gen_lens[i]))
                        eos_penalties[i]   = float(fi.get("eos_penalty",    eos_penalties[i]))
                        progress_ratios[i] = float(fi.get("progress_ratio", progress_ratios[i]))
                        perplexities[i]    = float(fi.get("perplexity",     perplexities[i]))
            # Save per-episode prompts for envs that finished on this step
            for i, done in enumerate(done_flags):
                if not done:
                    continue
                final_tokens = current_tokens[i]  # seed + all policy tokens for this env
                goal_i = batch_goals[i] if i < len(batch_goals) else batch_goals[0]
                gl_i   = int(gen_lens[i])
                _append_prompt_record(tokenizer, final_tokens, gl_i, goal_i, episode_idx + i, i)
   
            for i, done in enumerate(done_flags):
                if done:
                    logger.info(
                        "episode=%d env=%d gen_len=%s valid=%.3f reward=%.2f progress=%.3f",
                        episode_idx + i, i,
                        str(gen_lens[i]),
                        1.000,                          # or your true validity metric
                        float(rewards[i]),
                        float(progress_ratios[i]),
                    )
            for i in range(BATCH_SIZE):
                new_tokens = current_tokens[i] + [batch_actions[i]]
                transitions_batch[i].append(Transition(
                    obs=current_tokens[i][:],
                    goal=batch_goals[i],
                    action=batch_actions[i],
                    log_prob=batch_log_probs[i].detach(),
                    reward=rewards[i],
                    done=done_flags[i],
                    next_obs=new_tokens[:],
                    next_goal=batch_goals[i],
                    value=batch_values[i],
                    achieved_goal=len(new_tokens)
                ))
                # only update the prompt buffer if the episode isn’t done
                if not done_flags[i]:
                    current_tokens[i] = new_tokens

            dones = done_flags
            step_count += 1
            for j, (d, tok, g) in enumerate(zip(dones, current_tokens, batch_goals)):
                eff_goal = min(g, reward_tuning.MAX_NEW_GEN)
                body_len = len(tok) - prompt_len[j]
                if (not d) and body_len > eff_goal * 1.20:
                    dones[j] = True

            # -------------------------------------------------------------------

            if step_count > max(batch_goals) * 1.20:
                dones = [True] * BATCH_SIZE

        for i in range(BATCH_SIZE):
            seq = transitions_batch[i]
            if not seq:
                continue
            variants = her_relabel(seq, HER_FRACTION) 
            added_transitions = 0
            for tr, adv, ret in compute_gae(variants, gamma=GAMMA, lam=LAMBDA_GAE):
                replay_buffer.add((tr, adv, ret), abs(adv) + 1e-5)
                added_transitions += 1
            #logger.debug(
            #    "New Buffer Content: "
            #    f"{[(tr.obs, tr.goal, tr.reward) for (tr, _, _) in replay_buffer.buffer[-added_transitions:]]}"
            #)
        #logger.info(f"[BUFFER CHECK] Replay Buffer size before PPO update: {len(replay_buffer)}")
        # Check buffer content before PPO update
        #logger.info(f"[REPLAY BUFFER CONTENT] Current buffer size: {len(replay_buffer)}")
        #logger.debug(f"[REPLAY BUFFER] Content: {[(t.obs, t.goal, t.reward) for t, _, _ in replay_buffer.buffer]}")

        
        # Check if PPO update is triggered
        if len(replay_buffer) >= BATCH_SIZE * 2:
            logger.info(f"[PPO UPDATE TRIGGERED] Buffer size: {len(replay_buffer)}")
            log_flush()
            mini_batch, indices = replay_buffer.sample(
                min(BATCH_SIZE * 16, len(replay_buffer)), alpha=0.7
            )
            # Log replay buffer sample for diagnostics
            if len(replay_buffer) > 0:
                sample_obs = [t.obs for t, _, _ in replay_buffer.buffer[:5]]
                sample_rewards = [t.reward for t, _, _ in replay_buffer.buffer[:5]]
                logger.debug(f"Replay Buffer Sample (First 5): Obs: {sample_obs}, Rewards: {sample_rewards}")

            if mini_batch:
                #logger.debug(f"[SAMPLED TRANSITIONS] Sampled {len(mini_batch)} transitions. Indices: {indices}")
                #logger.debug(f"Sampled Transitions Details: {[(t.obs, t.goal, t.reward) for t, _, _ in mini_batch]}")
                logger.info(f"[PPO UPDATE] Mini-batch size: {len(mini_batch)}")

                LAST_100_REWARD_AVG = float(np.mean(all_rewards[-100:])) if all_rewards else 0.0
                # Apply PPO update
                ppo_update(
                    policy_net, value_net,
                    optimizer_policy, optimizer_value,
                    mini_batch,
                    scheduler_policy, scheduler_value,
                    eps_clip=EPS_CLIP, epochs=PPO_EPOCHS
                )

                if scheduler_policy is not None:
                    scheduler_policy.step()
                if scheduler_value is not None:
                    scheduler_value.step()

                # Update priorities in the replay buffer
                new_prios = [abs(adv) + 1e-5 for (_, adv, _) in mini_batch]
                replay_buffer.update_priorities(indices, new_prios)
            #else:
                #logger.debug("[PPO UPDATE] No transitions sampled, skipping PPO update.")
        # -------------------------------------------------------------------- #
        # helper – compute final episode statistics from its transition list
        # -------------------------------------------------------------------- #
        def _episode_stats(traj: list[Transition], goal_len: int):
            """
            Returns (gen_len, eos_penalty, progress_ratio).
            • gen_len      – number of tokens generated after the prompt
            • eos_penalty  – 0   if EOS was used, -0.05 otherwise
            • progress     – gen_len / goal_len  (clipped to [0, 1])
            """
            if not traj:                                    # safety guard
                return 0, 0.0, 0.0
        
            last = traj[-1]
            prompt_len0 = len(traj[0].obs) if traj and isinstance(traj[0].obs, list) else 0
            gen_len = last.achieved_goal - prompt_len0

            used_eos   = (last.action == EOS_TOKEN_ID)
            early_eos  = used_eos and gen_len < int(goal_len * 0.9)
            eos_pen    = -0.05 if early_eos else 0.0   # same sign as in compute_reward
            progress = min(gen_len / float(goal_len), 1.0)
            return gen_len, eos_pen, progress

        avg_batch_reward = np.mean([seq[-1].reward for seq in transitions_batch if seq])
        logger.info(f"[Batch {batch_idx}] Avg reward: {avg_batch_reward:.2f}")
        for i in range(BATCH_SIZE):
            seq = transitions_batch[i]
            # Adaptive length threshold only
            curr_len_thresh = min(8192, 64 + (episode_idx + i) * 32)
            logger.info(
                f"[THRESHOLD CHECK] Episode {episode_idx + i}: "
                f"Length Threshold = {curr_len_thresh}, Generated Length = {gen_lens[i]}"
            )
            if seq and seq[-1].achieved_goal >= curr_len_thresh:
                pass
            elif seq and seq[-1].achieved_goal >= 1024:
                logger.info(f"[BOOTSTRAP SAVE] len={seq[-1].achieved_goal}")
                prompt_tokens = original_prompts[i]
                prompt_str = tokenizer.decode(prompt_tokens, skip_special_tokens=True).strip()
                metadata = {
                    "prompt": prompt_str,
                    "gen_len": int(seq[-1].achieved_goal),
                    "reward": float(seq[-1].reward),
                    "validity": 1.0,
                }
                with open("training_prompts.jsonl", "a") as f_jsonl:
                    f_jsonl.write(json.dumps(metadata) + "\n")
                with open("training_prompts.txt", "a") as f_txt:
                    f_txt.write(prompt_str + "\n")
                            
            gen_len, eos_pen, progress = _episode_stats(transitions_batch[i],
                                                batch_goals[i])
            reward_i = transitions_batch[i][-1].reward if transitions_batch[i] else 0.0
        
            logger.info(
                f"[Episode {episode_idx + i}] "
                f"gen_len={gen_len}, eos_penalty={eos_pen:.2f}, "
                f"reward={reward_i:.2f}, progress={progress:.3f}"
            )
        
        for i in range(len(gen_lens)):
            gen_len   = gen_lens[i]
            validity = float(validities[i])
            eos_pen   = eos_penalties[i]
            progress  = progress_ratios[i]
            reward_i  = transitions_batch[i][-1].reward if transitions_batch[i] else 0.0
            ppl       = perplexities[i]
        
            logger.info(
                f"[Episode {episode_idx + i}] "
                f"gen_len={gen_len}, validity={validity:.3f}, "
                f"eos_penalty={eos_pen:.2f}, reward={reward_i:.2f}, "
                f"ppl={ppl:.2f}, "
                f"progress={progress:.3f}"
            )
            # --- NEW: force the message out right now -----------------
            for h in logger.handlers:
                if hasattr(h, "flush"):
                    h.flush()
            # ----------------------------------------------------------
            with open(TRACE_PATH, "a") as f:
                f.write(
                    f"episode={episode_idx + i:04d} env={i} "
                    f"gen_len={gen_len} valid={validity:.3f} "
                    f"reward={reward_i:.2f} progress={progress:.3f}\n"
                )

            # --- Length-based checkpoint: save as soon as we cross 8k tokens ---
            LEN_TRIGGER = 8000
            if gen_len >= LEN_TRIGGER and gen_len > BEST_GEN_LEN:
                BEST_GEN_LEN = gen_len
                len_tag = f"len{LEN_TRIGGER}plus_ep{episode_idx + i}_env{i}_len{gen_len}"
                save_checkpoint(
                    episode_idx + i,      # use this episode index for the ckpt metadata
                    best_reward,          # keep current best reward record
                    policy_net, value_net,
                    optimizer_policy, optimizer_value,
                    scheduler_policy, scheduler_value,
                    SAVE_PATH,
                    is_best=False,        # this is length-based, not reward-based
                    do_len_save=True,
                    len_save_tag=len_tag,
                )
                logger.info(
                    f"[LEN-CHECKPOINT] Saved policy at episode={episode_idx + i}, "
                    f"env={i}, gen_len={gen_len}, BEST_GEN_LEN={BEST_GEN_LEN}"
                )

            gen_lengths_list.append(gen_len)
            eos_penalties_list.append(eos_pen)
            all_rewards.append(reward_i)
            episodes_history.append(episode_idx + i)

        if len(replay_buffer) >= BATCH_SIZE * 2:
            mini_batch, indices = replay_buffer.sample(
                min(BATCH_SIZE * 20, len(replay_buffer)), alpha=0.6
            )
            LAST_100_REWARD_AVG = float(np.mean(all_rewards[-100:])) if all_rewards else 0.0
            ppo_update(
                policy_net, value_net,
                optimizer_policy, optimizer_value,
                mini_batch,
                scheduler_policy, scheduler_value,
                eps_clip=EPS_CLIP, epochs=PPO_EPOCHS
            )
            if scheduler_policy is not None:
                scheduler_policy.step()
            if scheduler_value is not None:
                scheduler_value.step()

            new_prios = [abs(adv) + 1e-5 for (_, adv, _) in mini_batch]
            replay_buffer.update_priorities(indices, new_prios)
        successes = sum(
            1 for seq in transitions_batch
            if seq and seq[-1].reward > 0
        )
        batch_success_rate = successes / BATCH_SIZE
        success_history.append(batch_success_rate)
        #scheduler_policy.step()
        #scheduler_value.step()
        current_ep = min(episode_idx + BATCH_SIZE, NUM_EPISODES)
        if current_ep % 10 == 0:
            avg_g = float(np.mean(gen_lengths_list[-50:])) if gen_lengths_list else 0.0
            avg_e = float(np.mean(eos_penalties_list[-50:])) if eos_penalties_list else 0.0
            adjust_reward_weights(avg_g, None, avg_e)  # validity unused now

            K = 10
            recent = success_history[-K:]
            if len(recent) == K:  # ensure enough data
                avg_success = sum(recent) / K
                new_her = update_her_fraction(avg_success, HER_FRACTION)
                logger.info(
                    f"Adjust HER: {HER_FRACTION:.3f} → {new_her:.3f} "
                    f"(success={avg_success:.2f})"
                )
                HER_FRACTION = new_her
        last_reward = max(
            (tr[-1].reward for tr in transitions_batch if tr),
            default=-1e9          
        )
        is_best = last_reward > best_reward
        if is_best:
            best_reward = last_reward
        
        # --- length-based tagging for the best-by-reward checkpoint ---
        LEN_THRESHOLD = 16000
        max_gen_in_batch = max(gen_lengths_list[-BATCH_SIZE:], default=0)
        is_len_over = (max_gen_in_batch >= LEN_THRESHOLD)
        len_save_tag = f"len{LEN_THRESHOLD}plus_{max_gen_in_batch}"

        # Save "best" checkpoint (unchanged)
        if is_best:
            save_checkpoint(current_ep, best_reward,
                            policy_net, value_net,
                            optimizer_policy, optimizer_value,
                            scheduler_policy, scheduler_value,
                            SAVE_PATH,
                            is_best=True,
                            do_len_save=is_len_over,
                            len_save_tag=len_tag)
        
        # ---- length cap ramp (every ~200 episodes) ----
        if current_ep % 50 == 0:  # adjust occasionally
            # Start low to make early credit assignment easy, then ramp.
            # Example schedule: 256 → 512 → 768 → 1024 → ... → 4096 (up to 16384)
            steps = current_ep // 200
            new_cap = min(16384, 256 + steps * 256)
            reward_tuning.set_max_new_gen(new_cap)

        # Decide whether to checkpoint this batch.
        # We want a checkpoint every CHECKPOINT_INTERVAL episodes (250)
        # regardless of BATCH_SIZE, plus one at the final episode.
        nominal_ep_marker = (current_ep // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
        
        should_save_final = (current_ep == NUM_EPISODES) and (current_ep > last_ckpt_episode)
        should_save_periodic = (
            not should_save_final and
            nominal_ep_marker > last_ckpt_episode and
            nominal_ep_marker > 0
        )
        
        # Periodic or final checkpoint, with 16k+ tagged copy if applicable
        if should_save_periodic or should_save_final:
            # For the very last checkpoint we label with the true final episode;
            # otherwise we snap to the interval boundary (…250, 500, 750, …).
            ckpt_ep = current_ep if should_save_final else nominal_ep_marker
        
            save_checkpoint(ckpt_ep, best_reward,
                            policy_net, value_net,
                            optimizer_policy, optimizer_value,
                            scheduler_policy, scheduler_value,
                            SAVE_PATH,
                            is_best=is_best,
                            do_len_save=is_len_over,
                            len_save_tag=len_tag)
        
            last_ckpt_episode = ckpt_ep
            logger.info(
                f"[CHECKPOINT] Saved checkpoint at Episode {ckpt_ep}. "
                f"Best Reward: {best_reward:.2f}"
            )
            log_flush()

        

        logger.info(
            f"[Checkpoint] Ep {current_ep}/{NUM_EPISODES}: "
            f"last_reward={all_rewards[-1]:.2f}, best={best_reward:.2f}"
        )
        
        if current_ep % 500 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(episodes_history, all_rewards, label="Episode Reward", alpha=0.7)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"RL-GOAL Training Progress (up to ep {current_ep})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(SAVE_PATH, f"reward_curve_ep{current_ep}.png"))
            plt.close()
    logger.info("Training complete. Best reward=%.3f", best_reward)
    logger.info("=== FINISHED RL-GOAL TRAIN ===")
    log_flush()
    return policy_net, value_net, all_rewards, episodes_history


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    for i in range(5):  # Run 5 test episodes
        run_episode(i)

    # Then start training
    policy_net, value_net, rewards_history, episodes_history = train_sota_multistep()

