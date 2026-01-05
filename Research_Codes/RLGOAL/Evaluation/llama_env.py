import gymnasium as gym
from gymnasium import spaces
import random
import torch
from reward_tuning import (
    get_tokenizer,
    compute_prompt_validity,
    w_length, w_prompt, compute_reward, MAX_NEW_GEN
)
import numpy as np
from gymnasium.envs.registration import register, registry
import collections
import reward_tuning
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)
MAX_CTX = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def log_flush():
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()
NUM_EPISODES = 500000
MAX_STEPS = 64
START_GOAL_RANGE = (64, 256)
END_GOAL_RANGE   = (8192, 16384)
try:
    gym.spec("LLaMAMultiStepEnv-v0")
except gym.error.Error:
    register(
        id="LLaMAMultiStepEnv-v0",
        entry_point="llama_env:LLaMAMultiStepEnv",
        max_episode_steps=MAX_STEPS,
    )


class LLaMAMultiStepEnv(gym.Env):
    _global_episode_counter = 0
    def __init__(self):
        super().__init__()
        self.tokenizer    = get_tokenizer()
        vocab_size        = self.tokenizer.vocab_size
        self.EOS_TOKEN_ID = self.tokenizer.eos_token_id
        # now define spaces
        self.action_space      = spaces.Discrete(vocab_size)
        self.max_ctx = MAX_CTX
        self.observation_space = spaces.Box(
            low=0, high=vocab_size - 1, shape=(self.max_ctx,), dtype=np.int32
        )
        self.episode_length = 0
        self.goal = None
        self.current_tokens = None
        self.past_lengths = collections.deque(maxlen=20)

    def _get_obs(self):
        """
        Build an observation consistent with observation_space.
        - If observation_space is Dict('tokens', 'goal'), return that Dict.
        - Otherwise return a 1D int32 array of tokens (padded to max_ctx).
        """
        import numpy as np
        max_ctx = getattr(self, "max_ctx", 1024)
        toks = np.array(self.current_tokens[-max_ctx:], dtype=np.int32)
        if toks.shape[0] < max_ctx:
            pad = np.zeros(max_ctx - toks.shape[0], dtype=np.int32)
            toks = np.concatenate([pad, toks], axis=0)
    
        # If your env uses a Dict space with 'tokens'/'goal', return a dict
        try:
            from gymnasium import spaces
            if isinstance(self.observation_space, spaces.Dict):
                return {"tokens": toks, "goal": int(self.goal)}
        except Exception:
            pass  # fall back to array
    
        # Otherwise, return the tokens array
        return toks
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment. Supports optional seeding and optional
        'base_prompt_tokens' and 'goal' overrides passed via Gymnasium's options.
        """
        super().reset(seed=seed)
    
        # episode bookkeeping
        self.episode_length = 0
        self.last_progress  = 0.0
        self.last_body_len  = 0
    
        # ---------- 1) choose goal ----------
        # Allow a caller-provided goal (broadcast for all sub-envs under SyncVectorEnv)
        if options and ("goal" in options) and options["goal"]:
            try:
                self.goal = int(options["goal"])
            except Exception:
                self.goal = None
    
        if self.goal is None:
            # Adaptive curriculum (same spirit as v14)
            if len(self.past_lengths) >= 5:
                avg_len = sum(self.past_lengths) / len(self.past_lengths)
                adaptive_start_min = max(64,  int(0.5 * avg_len))
                adaptive_start_max = max(128, int(0.8 * avg_len))
            else:
                adaptive_start_min, adaptive_start_max = START_GOAL_RANGE
    
            frac    = min(self._get_global_episode() / NUM_EPISODES, 1.0)
            min_len = int((1 - frac) * adaptive_start_min + frac * END_GOAL_RANGE[0])
            max_len = int((1 - frac) * adaptive_start_max + frac * END_GOAL_RANGE[1])
            self.goal = random.randint(min_len, max_len)
    
        # ---------- 2) seed prompt ----------
        tok = get_tokenizer()
        default_text = "The following text is generated:"
    
        if options and "base_prompt_tokens" in options and options["base_prompt_tokens"]:
            toks = [int(t) for t in options["base_prompt_tokens"] if int(t) > 2]
            self.current_tokens = toks[:] if len(toks) >= 3 else tok.encode(default_text, add_special_tokens=False)
        else:
            self.current_tokens = tok.encode(default_text, add_special_tokens=False)
    
        self.base_prompt_len = len(self.current_tokens)
    
        if not hasattr(self, "past_lengths"):
            from collections import deque
            self.past_lengths = deque(maxlen=32)
    
        obs  = self._get_obs()
        info = {"goal": int(self.goal), "progress_ratio": 0.0}
        return obs, info



    def step(self, action):
        """
        One environment step.
        Returns: observation, reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # --- 0) apply action ---
        self.episode_length += 1
        a = int(action)
        self.current_tokens.append(a)
    
        # --- 1) basic bookkeeping ---
        # policy-generated body (exclude fixed prefix), keep only non-special > 2
        body_len = max(0, len(self.current_tokens) - self.base_prompt_len)
        effective_goal = max(1, min(int(self.goal), MAX_NEW_GEN)) 
        progress_ratio = body_len / float(effective_goal)
    
        terminated = (a == self.EOS_TOKEN_ID)
        truncated  = (self.episode_length >= MAX_STEPS)
        done = terminated or truncated
    
        # overshoot guard (force EOS if we exceed 130% of effective goal)
        if (not done) and (body_len > int(1.30 * effective_goal)):
            self.current_tokens.append(self.EOS_TOKEN_ID)
            terminated = True
            done = True
    
        # --- 2) dense shaping (incremental) ---
        last_progress = getattr(self, "last_progress", 0.0)
        #incremental_reward = 0.05 * (progress_ratio - last_progress)
        incremental_reward =0.0
        # --- 3) prepare info early so it's always defined ---
        info = {
            "goal_len":        int(self.goal),
            "effective_goal":  int(effective_goal),
            "progress_ratio":  float(min(progress_ratio, 1.0)),
            "gen_length":      int(body_len),   # live body length each step
            "prompt_validity": 1.0,             # will be updated at done
            "eos_penalty":     0.0,
            "perplexity":      0.0,
            "used_eos":        bool(terminated),
        }
    
        # --- 4) default reward; will be updated on done ---
        reward = float(incremental_reward)
    
        # --- 5) update the prompt window for reward validity (last 10 non-special) ---
        filtered = [t for t in self.current_tokens if int(t) > 2]
        window = filtered[-10:]
        if len(window) >= 3:
            self.prompt_tokens_for_reward = window
        else:
            # fallback to fixed prefix if we don't have enough fresh tokens
            self.prompt_tokens_for_reward = self.current_tokens[: self.base_prompt_len]
    
        # --- 6) terminal reward computed on the victim model's continuation ---
        if done:
            used_eos = bool(terminated)
        
            # Use the FULL prompt (prefix + agent tokens) as conditioning
            final_reward, gen_length, prompt_validity, eos_penalty, perplexity = compute_reward(
                prompt_tokens=self.current_tokens,   # <— full prompt goes to the victim
                goal_len=self.goal,
                agent_gen_ids=None,                  # <— None => fall back to victim generation path
                used_eos=used_eos
            )
        
            reward = float(final_reward + incremental_reward)

            info.update({
                "goal_len": int(self.goal),
                "effective_goal": int(min(int(self.goal), reward_tuning.MAX_NEW_GEN, reward_tuning.get_boot_cap())),
                "gen_length": int(gen_length),
                "prompt_validity": float(prompt_validity),
                "eos_penalty": float(eos_penalty),
                "perplexity": float(perplexity),
                "used_eos": bool(used_eos),
                "progress_ratio": 1.0,
            })


    
            # keep a small history for curriculum
            try:
                self.past_lengths.append(int(gen_length)) 
            except Exception:
                pass
    
        # --- 7) persist rolling signals and return ---
        self.last_progress = progress_ratio
        self.last_body_len = body_len
    
        return self._get_obs(), reward, bool(terminated), bool(truncated), info



    def _get_global_episode(self):
        """Monotonic counter shared across *all* env instances."""
        LLaMAMultiStepEnv._global_episode_counter += 1
        return LLaMAMultiStepEnv._global_episode_counter
