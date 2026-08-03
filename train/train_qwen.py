import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)

MODEL  = os.environ.get("LP_MODEL", "Qwen/Qwen2.5-Omni-7B")
DATA   = os.environ.get("WS_DATA", os.path.join(_HERE, "..", "data/vggsound_train.parquet"))
OUT    = os.environ.get("WS_OUT", "/tmp/cs_adapter")
INIT   = os.environ.get("WS_INIT", "")
EPOCHS = int(os.environ.get("WS_EPOCHS", "1"))
LR     = float(os.environ.get("WS_LR", "5e-5"))
ACCUM  = int(os.environ.get("WS_ACCUM", "8"))
SEED   = int(os.environ.get("WS_SEED", "0"))
ANS_W  = float(os.environ.get("WS_ANS_WEIGHT", "10.0"))
NFRAMES = int(os.environ.get("VGG_NFRAMES", "8"))
VISUAL_SWAP_W = float(os.environ.get("WS_VISUAL_SWAP_WEIGHT", "1.0"))

import pandas as pd, torch, torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import LoraConfig, get_peft_model, TaskType

df = pd.read_parquet(DATA)
if VISUAL_SWAP_W != 1.0:
    answers = df["reward_model"].map(lambda x: str(x.get("ground_truth", "")).lower())
    is_visual_swap = df["extra_info"].map(
        lambda x: x.get("task") == "ADVH" and x.get("content_swapped") == "video"
    )
    n_total = len(df)
    if n_total % 2:
        raise SystemExit("WS_VISUAL_SWAP_WEIGHT requires an even sample count for exact yes/no balance")
    parts = []
    for offset, answer in enumerate(("yes", "no")):
        pool = df[answers == answer]
        weights = is_visual_swap.loc[pool.index].map(
            lambda flag: VISUAL_SWAP_W if flag else 1.0
        )
        parts.append(pool.sample(n=n_total // 2, replace=True, weights=weights,
                                 random_state=SEED + offset))
    df = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
else:
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"data={DATA} n={len(df)} epochs={EPOCHS} lr={LR} accum={ACCUM} seed={SEED} "
      f"init={INIT or 'fresh'} answer_weight={ANS_W} visual_swap_weight={VISUAL_SWAP_W}", flush=True)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda")
try: model.disable_talker()
except Exception: pass
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok = proc.tokenizer

_TM = os.environ.get("WS_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
_R = int(os.environ.get("WS_LORA_R", "32"))
_ALPHA = int(os.environ.get("WS_LORA_ALPHA", "16"))
print(f"[lora] r={_R} alpha={_ALPHA}", flush=True)
lora = LoraConfig(r=_R, lora_alpha=_ALPHA, target_modules=_TM.split(","), task_type=TaskType.CAUSAL_LM,
                  lora_dropout=0.0, bias="none")
model = get_peft_model(model, lora)
if INIT:
    from safetensors.torch import load_file
    sd = load_file(os.path.join(INIT, "adapter_model.safetensors"))
    P = dict(model.named_parameters()); n = 0
    for k, v in sd.items():
        kk = k.replace(".weight", ".default.weight")
        if kk in P:
            with torch.no_grad(): P[kk].copy_(v.to(P[kk].device, P[kk].dtype)); n += 1
    if n != len(sd):
        raise SystemExit(f"adapter init key mismatch: matched {n}/{len(sd)} from {INIT}")
    print(f"initialized LoRA from {INIT} ({n} tensors)", flush=True)
model.print_trainable_parameters()
th = model.base_model.model.thinker
try: th.enable_input_require_grads()
except Exception: pass
if int(os.environ.get("WS_GRAD_CKPT", "0")):
    th.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
else:
    print("gradient checkpointing OFF (set WS_GRAD_CKPT=1 to restore)", flush=True)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def build_inputs(row):
    sys_msg = next(m["content"] for m in row["prompt"] if m["role"] == "system")
    content = []
    if len(row["videos"]): content.append({"type": "video", "video": row["videos"][0], "nframes": NFRAMES})
    if len(row["audios"]): content.append({"type": "audio", "audio": row["audios"][0]})
    q = next(m["content"] for m in row["prompt"] if m["role"] == "user").replace("<video>", "").replace("<audio>", "").strip()
    content.append({"type": "text", "text": q})
    msgs = [{"role": "system", "content": [{"type": "text", "text": sys_msg}]},
            {"role": "user", "content": content}]
    prompt_text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + row["response"] + "<|im_end|>\n"
    audios, images, videos, vkw = process_mm_info(msgs, use_audio_in_video=False, return_video_kwargs=True)
    if isinstance(vkw.get("fps"), (list, tuple)) and len(vkw["fps"]): vkw = {**vkw, "fps": vkw["fps"][0]}
    full = proc(text=full_text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False, **vkw)
    p_only = proc(text=prompt_text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=False, **vkw)
    L_p = p_only["input_ids"].shape[1]
    return full, L_p


def decision_weights(labels_1d, w_dec):
    ids = labels_1d.tolist()
    w = torch.ones(len(ids), dtype=torch.float32, device=labels_1d.device)
    if w_dec == 1.0:
        return w
    acc, inside = "", False
    for j, t in enumerate(ids):
        if t == -100:
            continue
        acc += tok.decode([t])
        if inside:
            if acc.rstrip().endswith("</"):
                inside = False
            else:
                w[j] = w_dec
        elif acc.rstrip().endswith("<answer>"):
            inside = True
    return w


def prefetch(rows, workers=int(os.environ.get("WS_WORKERS", "4"))):
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    with ThreadPoolExecutor(workers) as ex:
        q, it = deque(), iter(rows)
        for r in it:
            q.append((r, ex.submit(build_inputs, r)))
            if len(q) >= workers * 2: break
        while q:
            r, fut = q.popleft()
            try: nxt = next(it)
            except StopIteration: pass
            else: q.append((nxt, ex.submit(build_inputs, nxt)))
            yield r, fut.result()


t0, step, loss_acc = time.time(), 0, 0.0
opt.zero_grad()
for ep in range(EPOCHS):
    for i, (row, (full, L_p)) in enumerate(prefetch([df.iloc[j] for j in range(len(df))])):
        full = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in full.items()}
        labels = full["input_ids"].clone()
        labels[:, :L_p] = -100
        out = model.base_model.model.thinker(input_ids=full["input_ids"],
                                             attention_mask=full.get("attention_mask"),
                                             **{k: v for k, v in full.items() if k not in ("input_ids", "attention_mask")})
        resp_logits = out.logits[:, L_p - 1:-1, :].float()
        resp_labels = labels[:, L_p:]
        ce = F.cross_entropy(resp_logits.reshape(-1, resp_logits.size(-1)),
                             resp_labels.reshape(-1), ignore_index=-100, reduction="none")
        w = decision_weights(resp_labels[0], ANS_W)
        loss = (ce * w).sum() / w.sum()
        (loss / ACCUM).backward()
        loss_acc += loss.item()
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); step += 1
            el = time.time() - t0
            eta = el / (i + 1 + ep * len(df)) * (EPOCHS * len(df)) - el
            print(f"  ep{ep} it{i+1}/{len(df)} step{step} loss={loss_acc/ACCUM:.4f} "
                  f"({el:.0f}s, eta {eta/60:.0f}m)", flush=True)
            loss_acc = 0.0
    opt.step(); opt.zero_grad()

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
print(f"\nsaved adapter -> {OUT}", flush=True)
import json
print("adapter_config:", json.load(open(os.path.join(OUT, "adapter_config.json")))
      if os.path.exists(os.path.join(OUT, "adapter_config.json")) else "MISSING")
