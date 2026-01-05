# infer_rl_goal.py
"""
To run no prefix baseline
python infer_rl_goal.py \
  --random_policy \
  --num_samples 1000 \
  --victim_model deepseek-ai/deepseek-coder-7b-base-v1.5 \
  --reward_mode victim \
  --goal 16384 \
  --seed 42 \
  --max_steps 0 \
  --csv baseline_noprefix_ds.csv \
  --chunk_index $PBS_ARRAY_INDEX \
  --num_chunks 15

To run using policy attacker
 python infer_rl_goal.py \
   --ckpt len8000plus_ep62944_env0_len16377_ep62944.pt \
   --num_samples 1000 \
   --victim_model EleutherAI/gpt-neo-1.3B \
   --reward_mode victim \
   --goal 16384 \
   --seed 42 \
   --max_steps 64 \
   --csv policy_EL1B.csv \
   --chunk_index $PBS_ARRAY_INDEX \
   --num_chunks 15

To run random policy
python infer_rl_goal.py \
  --random_policy \
  --num_samples 1000 \
  --victim_model meta-llama/Llama-2-7b-hf \
  --reward_mode victim \
  --goal 16384 \
  --seed 42 \
  --max_steps 64 \
  --csv random_policy.csv \
  --chunk_index $PBS_ARRAY_INDEX \
  --num_chunks 15
"""
import os, math, torch, argparse, time, random
import transformers
from transformers import AutoTokenizer
from RL_GOAL import MiniPolicyTransformer  # uses same arch as training
from reward_tuning import MAX_NEW_GEN, compute_reward, set_max_new_gen, set_boot_cap
import reward_tuning as rt
from transformers import AutoConfig
os.environ.setdefault("INFER_MODE", "1")
os.environ.setdefault("RL_GOAL_SKIP_MAIN_MODEL_LOAD", "1")
# Match training caps for victim LLM
set_max_new_gen(16384)
set_boot_cap(16384)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")  # must match training
torch.set_grad_enabled(False)

TP_THRESHOLDS = (512, 1024)

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_context_window_from_model_id(model_id: str) -> int:
    cfg = AutoConfig.from_pretrained(model_id)

    for attr in ("max_position_embeddings",):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val

    raise ValueError(
        f"{model_id}: config has no valid max_position_embeddings; "
        "set CONTEXT_WINDOW manually or extend the attribute list."
    )



def compute_tail_persistence(new_tokens, k_gram=8):
    """
    Tail-persistence metric from the EA pipeline. Here we define it for
    completeness, but in this script we only have access to the *length*
    of the victim continuation, not its token sequence, so we do not
    call this yet.
    """
    if len(new_tokens) < k_gram * 2:
        return {f"TP@{n}": 0 for n in TP_THRESHOLDS}, 0
    seen, last_novel_idx = set(), -1
    for i in range(len(new_tokens) - k_gram + 1):
        gram = tuple(new_tokens[i:i + k_gram])
        if gram not in seen:
            seen.add(gram)
            last_novel_idx = i + k_gram - 1
    tail_len = max(0, len(new_tokens) - (last_novel_idx + 1))
    tp_flags = {f"TP@{n}": int(tail_len >= n) for n in TP_THRESHOLDS}
    return tp_flags, tail_len


def attack_side_metrics(gen_length: int,
                        context_window: int,
                        cap_hit: bool,
                        latency_sec: float | None = None):
    """
    Attack-side metrics based only on the continuation *length*.

    We cannot see the victim's actual token sequence here (only its
    length via compute_reward), so we leave tail-persistence metrics
    at 0 and focus on OGF + stall + latency.
    """
    ogf = gen_length / max(1, context_window)
    stall = int(cap_hit)

    return {
        "OGF": ogf,
        "stall": stall,
        "tail_len": 0,
        "TP@512": 0,
        "TP@1024": 0,
        "latency_sec": latency_sec,
    }

def load_policy(ckpt_path: str) -> tuple[MiniPolicyTransformer, AutoTokenizer, int]:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    eos_id = tok.eos_token_id
    # match training dims
    policy = MiniPolicyTransformer(hidden_dim=256, n_layers=2, n_heads=4).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, tok, eos_id

class RandomPolicy:
    """
    Random goal-conditioned policy for ablation.

    It has the same call interface as MiniPolicyTransformer:
        logits = policy(ctx_tokens, goal_len, return_hidden=False)
    but returns uniform logits over the vocab (no learning).
    """
    def __init__(self, vocab_size: int, device: str = DEVICE):
        self.vocab_size = vocab_size
        self.device = device

    def __call__(self, ctx_tokens: torch.Tensor, goal_len: int, return_hidden: bool = False):
        # Uniform logits -> softmax(logits) is uniform over the vocab.
        return torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)


def load_random_policy() -> tuple[RandomPolicy, AutoTokenizer, int]:
    """
    Build a random goal-conditioned policy with the same tokenizer / vocab
    and eos id as RL-GOAL, but no learned parameters.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    eos_id = tok.eos_token_id
    vocab_size = tok.vocab_size
    policy = RandomPolicy(vocab_size=vocab_size, device=DEVICE)
    return policy, tok, eos_id
    
def top_k_top_p_filter(logits: torch.Tensor, top_k: int | None, top_p: float | None):
    # logits: (vocab,)
    scores = logits.clone()
    if top_k is not None and top_k > 0:
        kth = torch.topk(scores, k=min(top_k, scores.numel()))[0][-1]
        scores[scores < kth] = -float("inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = torch.softmax(scores, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cut = cum > top_p
        if cut.any():
            first_cut = torch.argmax(cut.to(torch.int32)).item()
            keep = sorted_idx[:first_cut+1]
            mask = torch.full_like(scores, -float("inf"))
            mask[keep] = scores[keep]
            scores = mask
    return scores

def generate_with_policy(
    policy,
    tok,
    prompt: str,
    goal_len: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    max_steps: int = 64,
    max_ctx: int = 1024,
    english_only: bool = False,
    gate_alpha: float = 0.0,
) -> tuple[str, list[int]]:
    """
    Use the trained RL policy to generate a short prefix (≤ max_steps tokens)
    given a base prompt and goal length. This mirrors the training setup where
    the agent only produced up to MAX_STEPS actions and the victim model did
    the long continuation.
    """
    tokens   = tok.encode(prompt, add_special_tokens=False)
    base_len = len(tokens)
    goal_len = int(goal_len)
    cap      = max(1, min(goal_len, MAX_NEW_GEN))   # overall new-token cap
    eos_id   = tok.eos_token_id

    # Only let the policy act for at most max_steps, like in training.
    for step in range(min(max_steps, cap)):
        ctx   = tokens[-max_ctx:]
        ctx_t = torch.tensor(ctx, dtype=torch.long, device=DEVICE)

        logits = policy(ctx_t, goal_len, return_hidden=False)  # (vocab,)
        logits = logits / max(temperature, 1e-6)

        # Optional: restrict to (basic) English-ish token set if you have a mask
        # if english_only and hasattr(policy, "english_mask"):
        #     logits[~policy.english_mask] = -float("inf")

        # Length-gating for EOS (if gate_alpha > 0).
        gen_len    = len(tokens) - base_len
        allow_after = int(max(0.0, min(gate_alpha, 1.0)) * cap)
        if gen_len < allow_after:
            logits[eos_id] = float("-inf")

        # top-k / nucleus
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

        # sample
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

        # skip specials 0/1/2, with simple resample fallback
        if action <= 2:
            for _ in range(4):
                action = dist.sample().item()
                if action > 2:
                    break
            if action <= 2:
                action = 3  # force a printable token

        tokens.append(action)
        if action == eos_id:
            break

    new_tokens = tokens[base_len:]
    text       = tok.decode(new_tokens, skip_special_tokens=True)
    return text, new_tokens

def run_single_attack(policy, tok, eos, args, victim_tok, run_idx: int = 0):
    """
    Run one policy attack, compute reward and attack-side metrics,
    and print a compact stats line. Returns a dict suitable for CSV
    plus the policy prefix text.
    """
    # 1) Use policy to craft a short prefix (≤ max_steps)
    text, new_ids = generate_with_policy(
        policy,
        tok,
        prompt=args.prompt,
        goal_len=args.goal,
        temperature=args.temp,
        top_k=(args.top_k or None),
        top_p=args.top_p,
        max_steps=args.max_steps,
        gate_alpha=args.gate_alpha,
    )

    # 2) Build base + full prompt token ids
    base_prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    full_prompt_ids = base_prompt_ids + new_ids

    # 3) Compute reward + measure latency
    if args.reward_mode == "victim":
        t0 = time.time()
    
        # If victim tokenizer differs from policy tokenizer, bridge via text:
        # decode with policy tok -> re-encode with victim tok
        if args.victim_model == MODEL_NAME:
            victim_prompt_ids = full_prompt_ids
        else:
            prompt_text = tok.decode(full_prompt_ids, skip_special_tokens=True)
            victim_prompt_ids = victim_tok.encode(prompt_text, add_special_tokens=False)
    
        reward, gen_len, validity, eos_pen, ppl = compute_reward(
            prompt_tokens=victim_prompt_ids,
            goal_len=args.goal,
            agent_gen_ids=None,
            used_eos=(len(new_ids) > 0 and new_ids[-1] == eos),
        )
        latency = time.time() - t0
        cap_hit = (gen_len >= MAX_NEW_GEN)

    else:
        t0 = time.time()
        # Score only the policy's own text (no victim call).
        reward, gen_len, validity, eos_pen, ppl = compute_reward(
            prompt_tokens=base_prompt_ids,
            goal_len=args.goal,
            agent_gen_ids=new_ids,
            used_eos=(len(new_ids) > 0 and new_ids[-1] == eos),
        )
        latency = time.time() - t0
        # No victim budget in this mode; treat as non-stall.
        cap_hit = False

    # 4) Attack-side metrics from length only
    ctx_window = args.context_window
    metrics = attack_side_metrics(
        gen_length=gen_len,
        context_window=ctx_window,
        cap_hit=cap_hit,
        latency_sec=latency,
    )
    # Success flags matching EA / baseline metrics
    success_ogf1 = int(metrics["OGF"] >= 1.0)
    success_ogf2 = int(metrics["OGF"] >= 2.0)
    success_ogf4 = int(metrics["OGF"] >= 4.0)

    # 5) Pretty-print (avoid spam if num_samples > 1)
    if run_idx == 0:
        print("\n=== POLICY PREFIX (truncated) ===")
        print(text)

        full_prompt_text = tok.decode(full_prompt_ids, skip_special_tokens=True)
        print("\n=== FULL ATTACK PROMPT (truncated) ===")
        print(full_prompt_text[:800])

    label = "victim_gen_len" if args.reward_mode == "victim" else "policy_gen_len"
    print(
        f"[Run {run_idx}] [Stats] {label}={gen_len} "
        f"reward={reward:.3f} validity={validity:.3f} eos_pen={eos_pen:.2f} "
        f"ppl≈{ppl:.1f} OGF={metrics['OGF']:.3f} stall={metrics['stall']} "
        f"latency={metrics['latency_sec']:.2f}s"
    )

    record = {
        # identity / reproducibility
        "run": int(run_idx),
        "base_seed": int(args.seed),
        "chunk_index": int(args.chunk_index),
        "num_chunks": int(args.num_chunks),
        "ckpt": str(args.ckpt) if not getattr(args, "random_policy", False) else "RANDOM_POLICY",
        "victim_model": str(args.victim_model),
        "goal": int(args.goal),
        "max_steps": int(args.max_steps),
        "temp": float(args.temp),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "gate_alpha": float(args.gate_alpha),
        "reward_mode": str(args.reward_mode),
        "context_window": int(args.context_window),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,

        # metrics
        "gen_len": int(gen_len),
        "OGF": float(metrics["OGF"]),
        "stall": int(metrics["stall"]),
        "tail_len": int(metrics["tail_len"]),
        "TP@512": int(metrics["TP@512"]),
        "TP@1024": int(metrics["TP@1024"]),
        "latency_sec": float(metrics["latency_sec"]) if metrics["latency_sec"] is not None else None,
        "success_ogf1": int(metrics["OGF"] >= 1.0),
        "success_ogf2": int(metrics["OGF"] >= 2.0),
        "success_ogf4": int(metrics["OGF"] >= 4.0),
        "reward": float(reward),
        "validity": float(validity),
        "eos_pen": float(eos_pen),
        "ppl": float(ppl),
    }

    return record, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./rl_goal/len8000plus_ep56105_env1_len16291_ep56105.pt")
    ap.add_argument("--prompt", default="The following text is generated:")
    ap.add_argument("--goal", type=int, default=16384)
    ap.add_argument("--victim_model", type=str,default=MODEL_NAME)
    ap.add_argument("--out", default=None)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=64,
        help="Max number of tokens the policy itself may emit (like MAX_STEPS in training).",
    )
    ap.add_argument(
        "--gate_alpha",
        type=float,
        default=0.0,
        help="Allow EOS after alpha*cap. Use 0.0 to disable gating; 0.8 for loose; 0.97 default.",
    )
    ap.add_argument(
        "--reward_mode",
        choices=["victim", "policy"],
        default="victim",
        help=(
            "'victim' = use victim to generate continuation (matches training); "
            "'policy' = score policy's own text directly."
        ),
    )
    # NEW: sampling + logging options
    ap.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of independent policy samples to draw.",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write a CSV with run-level stats (OGF, stall, etc.).",
    )
    ap.add_argument(
        "--context_window",
        type=int,
        default=None,
        help="Nominal context window used to compute OGF. If not set, derived from --victim_model config.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for sampling.",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="If set, enable PyTorch deterministic modes (may reduce speed).",
    )

    ap.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="Index of this job in a chunked evaluation (0-based).",
    )
    ap.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Total number of chunks/jobs sharing the same checkpoint and num_samples.",
    )
    ap.add_argument(
        "--random_policy",
        action="store_true",
        help="If set, use a random goal-conditioned policy instead of a trained RL checkpoint.",
    )
    args = ap.parse_args()
    # Base seeding for any initialization prior to per-run seeding
    set_all_seeds(args.seed)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Derive nominal context window from victim model config (no hardcoded default)
    derived_ctx = get_context_window_from_model_id(args.victim_model)
    if args.context_window is None:
        args.context_window = derived_ctx

    # Configure victim model for reward_tuning (independent of policy MODEL_NAME)
    rt.MODEL_NAME = args.victim_model
    rt.tokenizer = None
    rt.llama_model = None
    
    # Tokenizer used only for re-encoding prompts when victim differs from policy tokenizer
    victim_tok = AutoTokenizer.from_pretrained(args.victim_model)
    victim_tok.padding_side = "left"

    # 1) Load policy + tokenizer once
    if args.random_policy:
        print("[LOG] Using RandomPolicy baseline (no checkpoint).")
        policy, tok, eos = load_random_policy()
    else:
        policy, tok, eos = load_policy(args.ckpt)

    # 2) Run multiple attacks (chunked across jobs)
    total = args.num_samples
    num_chunks = max(1, args.num_chunks)
    chunk_index = max(0, min(args.chunk_index, num_chunks - 1))
    chunk_size = math.ceil(total / num_chunks)
    start = chunk_index * chunk_size
    end = min(total, (chunk_index + 1) * chunk_size)
    # Avoid clobbering --out across PBS array jobs
    if args.out and num_chunks > 1:
        root, ext = os.path.splitext(args.out)
        if not ext:
            ext = ".txt"
        args.out = f"{root}_chunk{chunk_index:02d}_of_{num_chunks:02d}{ext}"

    print(
        f"[LOG] Chunk {chunk_index+1}/{num_chunks}: "
        f"running runs {start}..{max(start, end-1)} of {total} total"
    )

    results = []
    last_text = ""
    completed_runs = set()
    # If running as a PBS/Slurm array, default to one CSV per chunk to avoid
    # concurrent writes to the same file (simpler + more robust on HPC).
    if args.csv and num_chunks > 1:
        root, ext = os.path.splitext(args.csv)
        if not ext:
            ext = ".csv"
        args.csv = f"{root}_chunk{chunk_index:02d}_of_{num_chunks:02d}{ext}"

    # CSV schema (must match keys in `record` from run_single_attack)
    fieldnames = [
        # identity / reproducibility
        "run",
        "base_seed",
        "chunk_index",
        "num_chunks",
        "ckpt",
        "victim_model",
        "goal",
        "max_steps",
        "temp",
        "top_p",
        "top_k",
        "gate_alpha",
        "reward_mode",
        "context_window",
        "torch_version",
        "transformers_version",

        # metrics
        "gen_len",
        "OGF",
        "stall",
        "tail_len",
        "TP@512",
        "TP@1024",
        "latency_sec",
        "success_ogf1",
        "success_ogf2",
        "success_ogf4",
        "reward",
        "validity",
        "eos_pen",
        "ppl",
    ]

    def append_csv_row(csv_path: str, row: dict):
        """Append one row, writing header if needed. Uses flock and fsync for HPC safety."""
        import csv
        try:
            import fcntl
        except ImportError:
            fcntl = None

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "a+", newline="") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)

            f.seek(0, os.SEEK_END)
            write_header = (f.tell() == 0)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

            # critical for resume correctness on HPC (avoid buffered loss on crash)
            f.flush()
            os.fsync(f.fileno())

            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)

    if args.csv is not None and os.path.exists(args.csv):
        import csv
        with open(args.csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "run" in reader.fieldnames:
                for row in reader:
                    try:
                        completed_runs.add(int(row["run"]))
                    except Exception:
                        pass

    appended = 0
    for i in range(start, end):
        # make runs stochastic but reproducible across all chunks
        if i in completed_runs:
            continue

        set_all_seeds(args.seed + i)
        record, text = run_single_attack(policy, tok, eos, args, victim_tok, run_idx=i)

        if args.csv is not None:
            append_csv_row(args.csv, record)
            completed_runs.add(i)
            appended += 1
        else:
            results.append(record)

        last_text = text

    if args.csv is not None:
        print(f"\n[LOG] Appended {appended} runs to {args.csv}")



    # 4) Keep old behaviour: write last policy prefix to file
    if args.out:
        with open(args.out, "w") as f:
            f.write(last_text)



if __name__ == "__main__":
    main()
