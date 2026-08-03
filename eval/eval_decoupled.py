import os, sys, json, re, time, contextlib
import pandas as pd, numpy as np, torch
from safetensors.torch import load_file
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import LoraConfig, get_peft_model, TaskType

MODEL      = os.environ.get("CE_MODEL", "Qwen/Qwen2.5-Omni-7B")
ADAPTER    = os.environ.get("CE_ADAPTER", "")
ADAPTER2   = os.environ.get("CE_ADAPTER2", "")
EVAL       = os.environ.get("CE_EVAL", "")
MEDIA_ROOT = os.environ.get("CE_MEDIA_ROOT", "")
OUT        = os.environ.get("CE_OUT", "/tmp/decoupled_eval.json")
RAW        = os.environ.get("CE_RAW", OUT.replace(".json", "_raw.jsonl"))
TAG        = os.environ.get("CE_TAG", "exp1a")
LIMIT      = int(os.environ.get("CE_LIMIT", "0"))
MAXNEW     = int(os.environ.get("CE_MAXNEW", "128"))
ALWAYS_ON  = os.environ.get("CE_ALWAYS_ON", "") == "1"
ECE_BINS   = int(os.environ.get("CE_ECE_BINS", "15"))
NFRAMES    = int(os.environ.get("VGG_NFRAMES", "8"))
ALPHA, R   = 16.0, 32.0

_ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)
_BARE_RE   = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_CONF_RE   = re.compile(r"<conf>\s*([01](?:\.\d+)?|\.\d+)\s*</conf>", re.IGNORECASE)
def parse_response(text, default_conf=0.5):
    if not text: return (None, default_conf, False, False)
    parsed = True
    m = _ANSWER_RE.search(text)
    if m is None:
        m = _BARE_RE.search(text); parsed = False
    if m is None: return (None, default_conf, False, False)
    ans = m.group(1).lower()
    conf, has_conf = default_conf, False
    cm = _CONF_RE.search(text)
    if cm is not None:
        try: conf = min(1.0, max(0.0, float(cm.group(1)))); has_conf = True
        except ValueError: conf = default_conf
    return (ans, conf, has_conf, parsed)
def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels).astype(bool)
    n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0: return None
    order = s.argsort(kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1; i = j + 1
    R1 = ranks[y].sum(); return float((R1 - n1 * (n1 + 1) / 2) / (n1 * n0))

def remap(p):
    if not MEDIA_ROOT or not p: return p
    for mk in ("/AVHBench/", "/CMM/", "/VGGSound/"):
        i = p.find(mk)
        if i >= 0: return MEDIA_ROOT.rstrip("/") + p[i + len(mk) - 1:]
    return p

df = pd.read_parquet(EVAL)
df["task"] = df["extra_info"].apply(lambda x: x.get("task"))
_avh = df["task"].isin(["ADVH", "VDAH", "AVM"])
if _avh.any(): df = df[_avh]
df = df.reset_index(drop=True)
if LIMIT:
    df = df.groupby("task", group_keys=False).head(max(1, LIMIT // df["task"].nunique())).reset_index(drop=True)
print(f"[data] EVAL={EVAL} n={len(df)}", flush=True)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda")
def merge_into_base(adapter):
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    sc = cfg.get("lora_alpha", ALPHA) / cfg.get("r", R)
    ad = load_file(os.path.join(adapter, "adapter_model.safetensors")); mods = {}
    for k, v in ad.items():
        kk = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        if kk.endswith(".lora_A.weight"): mods.setdefault(kk[:-14], {})["A"] = v
        elif kk.endswith(".lora_B.weight"): mods.setdefault(kk[:-14], {})["B"] = v
    P = dict(model.named_parameters()); n = 0
    for pf, d in mods.items():
        wk = pf + ".weight"
        if wk in P and "A" in d and "B" in d:
            with torch.no_grad(): P[wk].add_(((d["B"].float() @ d["A"].float()) * sc).to(P[wk].device, P[wk].dtype)); n += 1
    print(f"[merge-grounding] {n} modules scale={sc}", flush=True)
merge_into_base(ADAPTER)
if ADAPTER2:
    _TM = os.environ.get("WS_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    _c2 = os.path.join(ADAPTER2, "adapter_config.json")
    _cfg2 = json.load(open(_c2)) if os.path.exists(_c2) else {}
    _r2, _a2 = int(_cfg2.get("r", R)), int(_cfg2.get("lora_alpha", ALPHA))
    print(f"[conf-LoRA] r={_r2} alpha={_a2}", flush=True)
    model = get_peft_model(model, LoraConfig(r=_r2, lora_alpha=_a2, target_modules=_TM.split(","),
                                             task_type=TaskType.CAUSAL_LM, lora_dropout=0.0, bias="none"))
    sd = load_file(os.path.join(ADAPTER2, "adapter_model.safetensors")); P = dict(model.named_parameters()); n = 0
    for k, v in sd.items():
        kk = k.replace(".weight", ".default.weight")
        if kk in P:
            with torch.no_grad(): P[kk].copy_(v.to(P[kk].device, P[kk].dtype)); n += 1
    if n != len(sd): raise SystemExit(f"conf-LoRA load mismatch {n}/{len(sd)}")
    print(f"[conf-LoRA] loaded {n} tensors ({'always on' if ALWAYS_ON else 'toggled at <conf>'})", flush=True)
    THINKER = model.base_model.model.thinker
else:
    print("[conf-LoRA] none: grounding-only run", flush=True)
    THINKER = model.thinker
try: model.disable_talker()
except Exception: pass
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL, trust_remote_code=True)

def build(row):
    sm = next(m["content"] for m in row["prompt"] if m["role"] == "system")
    q = next(m["content"] for m in row["prompt"] if m["role"] == "user").replace("<video>", "").replace("<audio>", "").strip()
    c = []
    if len(row["videos"]): c.append({"type": "video", "video": remap(row["videos"][0]), "nframes": NFRAMES})
    if len(row["audios"]): c.append({"type": "audio", "audio": remap(row["audios"][0])})
    c.append({"type": "text", "text": q})
    return [{"role": "system", "content": [{"type": "text", "text": sm}]}, {"role": "user", "content": c}]

def encode(text, audios, images, videos):
    return proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                padding=True, use_audio_in_video=False).to("cuda")

def generate(inp, disable, max_new, keep_cache=False):
    ctx = model.disable_adapter() if (disable and ADAPTER2) else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        o = model.generate(**inp, max_new_tokens=max_new, do_sample=False, num_beams=1,
                           return_audio=False, use_audio_in_video=False,
                           return_dict_in_generate=True, use_cache=True)
    seq = o.sequences if hasattr(o, "sequences") else (o[0] if isinstance(o, tuple) else o)
    return seq, (getattr(o, "past_key_values", None) if keep_cache else None)

def conf_cut(gen_ids):
    n_ans = None
    for t in range(gen_ids.shape[0]):
        txt = proc.tokenizer.decode(gen_ids[:t + 1], skip_special_tokens=True)
        if "<conf>" in txt: return t + 1, ""
        if n_ans is None and "</answer>" in txt: n_ans = t + 1
    return (n_ans, "\n<conf>") if n_ans is not None else (gen_ids.shape[0], "\n<conf>")

KV_OK = True
DECODE_MODE = None

def continue_conf(seq, cache, plen, r_off, base_text, mm):
    global KV_OK, DECODE_MODE
    if KV_OK and cache is not None and hasattr(cache, "crop"):
        try:
            keep, extra = conf_cut(seq[0, plen:])
            ids = seq[:, :plen + keep]
            if extra:
                ex = proc.tokenizer(extra, add_special_tokens=False,
                                    return_tensors="pt")["input_ids"].to(ids.device)
                ids = torch.cat([ids, ex], dim=1)
            cache.crop(min(plen + keep, ids.shape[1] - 1))
            with torch.no_grad():
                out = THINKER.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       past_key_values=cache, use_cache=True,
                                       max_new_tokens=12, do_sample=False, num_beams=1)
            if DECODE_MODE is None:
                DECODE_MODE = "kv-reuse"
                print("[decode] kv-reuse: context retained, adapter switched on at <conf>", flush=True)
            return proc.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
        except Exception as e:
            KV_OK = False
            print(f"[decode] kv-reuse unavailable ({type(e).__name__}: {e}); re-encoding the prefix",
                  flush=True)
    if DECODE_MODE is None:
        DECODE_MODE = "reencode"
    if "<conf>" in r_off:      prefix = r_off[:r_off.index("<conf>") + len("<conf>")]
    elif "</answer>" in r_off: prefix = r_off[:r_off.index("</answer>") + len("</answer>")] + "\n<conf>"
    else:                      prefix = r_off.rstrip() + "\n<conf>"
    inp2 = encode(base_text + prefix, *mm)
    seq2, _ = generate(inp2, disable=False, max_new=12)
    return proc.batch_decode(seq2[:, inp2["input_ids"].shape[1]:], skip_special_tokens=True)[0]

done = 0
if os.path.exists(RAW):
    with open(RAW) as f: done = sum(1 for _ in f)
    print(f"[resume] {done} already in {RAW}", flush=True)
raw_f = open(RAW, "a")
recs = []
if done:
    with open(RAW) as f:
        for line in f:
            if line.strip(): recs.append(json.loads(line))

t0 = time.time()
for i, row in df.iterrows():
    if i < done: continue
    gt = str(row["reward_model"]["ground_truth"]).lower()
    msgs = build(row)
    base_text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    mm = process_mm_info(msgs, use_audio_in_video=False)
    inp = encode(base_text, *mm)
    plen = inp["input_ids"].shape[1]
    seq, cache = generate(inp, disable=not ALWAYS_ON, max_new=MAXNEW,
                          keep_cache=bool(ADAPTER2) and not ALWAYS_ON)
    r_off = proc.batch_decode(seq[:, plen:], skip_special_tokens=True)[0]
    ans, conf, has_conf, parsed = parse_response(r_off)
    r_conf = ""
    if ADAPTER2 and not ALWAYS_ON:
        r_conf = continue_conf(seq, cache, plen, r_off, base_text, mm)
        cm = _CONF_RE.search("<conf>" + r_conf)
        conf = min(1.0, max(0.0, float(cm.group(1)))) if cm else 0.5
        has_conf = cm is not None
    rec = dict(task=row["task"], gt=gt, resp=r_off, ans=ans,
               conf=float(conf), has_conf=bool(has_conf), parsed=bool(parsed), conf_gen=r_conf[:20])
    recs.append(rec); raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); raw_f.flush()
    if i % 200 == 0 or i == len(df) - 1:
        print(f"  [{i+1}/{len(df)}] {(time.time()-t0)/60:.0f}m elapsed", flush=True)
raw_f.close()

def metrics_for(rs):
    n = len(rs)
    corr = sum(int(r["ans"] == r["gt"]) for r in rs)
    comm = [r for r in rs if r["ans"] in ("yes", "no")]
    cc = [int(r["ans"] == r["gt"]) for r in comm]; cf = [r["conf"] for r in comm]
    ece = 0.0
    if comm:
        cfa = np.array(cf); coa = np.array(cc)
        idx = np.clip((cfa * ECE_BINS).astype(int), 0, ECE_BINS - 1)
        for b in range(ECE_BINS):
            m = idx == b
            if m.sum(): ece += (m.sum() / len(cfa)) * abs(coa[m].mean() - cfa[m].mean())
    au = auroc(cf, [c > 0.5 for c in cc]) if (comm and len(set(cc)) > 1) else None
    return dict(n=n, n_committed=len(comm), acc=round(corr / n, 4) if n else None,
                expr_auroc=(round(au, 4) if au is not None else None),
                ece=round(float(ece), 4) if comm else None,
                has_conf_rate=round(sum(int(r["has_conf"]) for r in rs) / n, 4) if n else None)
task_names = list(dict.fromkeys(r["task"] for r in recs))
per = {t: metrics_for([r for r in recs if r["task"] == t]) for t in task_names}
overall = metrics_for(recs)
def avg(keys, metric):
    vals = [per[t][metric] for t in keys if t in per and per[t].get(metric) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
variant = "grounding" if not ADAPTER2 else ("always-on" if ALWAYS_ON else "decoupled")
res = dict(tag=TAG, model=MODEL, adapter=ADAPTER, adapter2=ADAPTER2, eval=EVAL, n=len(recs),
           variant=variant, decode_mode=DECODE_MODE, ece_bins=ECE_BINS,
           per_task=per, overall=overall,
           avg_3task={k: avg(["ADVH", "VDAH", "AVM"], k) for k in ("acc", "expr_auroc", "ece")})
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n=== {variant.upper()} RESULT (decode={DECODE_MODE}) ===")
print(json.dumps({"overall": overall, "per_task": per, "avg_3task": res["avg_3task"]}, indent=2))
