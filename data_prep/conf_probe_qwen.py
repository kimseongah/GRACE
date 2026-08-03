import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..")); sys.path.insert(0, _HERE)

W = os.path.abspath(os.path.join(_HERE, ".."))
MODEL = os.environ.get("PC_MODEL", "Qwen/Qwen2.5-Omni-7B")
ADAPTER = os.environ.get("PC_ADAPTER", f"{W}/checkpoints/qwen_grounding/adapter")
DATA = os.environ.get("PC_DATA", f"{W}/data/vggsound_train.parquet")
OUT = os.environ.get("PC_OUT", f"{W}/data/qwen_probe.json")
NFRAMES = int(os.environ.get("VGG_NFRAMES", "8"))
ALPHA, R = 16.0, 32.0

import pandas as pd, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from safetensors.torch import load_file

df = pd.read_parquet(DATA)
print(f"data={DATA} n={len(df)} adapter={ADAPTER}", flush=True)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cuda")
if ADAPTER:
    _cfgp = os.path.join(ADAPTER, "adapter_config.json")
    if os.path.exists(_cfgp):
        _cfg = json.load(open(_cfgp)); _a, _r = float(_cfg.get("lora_alpha", ALPHA)), float(_cfg.get("r", R))
    else:
        _a, _r = ALPHA, R
    ad = load_file(os.path.join(ADAPTER, "adapter_model.safetensors")); sc, mods = _a / _r, {}
    for k, v in ad.items():
        kk = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        if kk.endswith(".lora_A.weight"): mods.setdefault(kk[:-14], {})["A"] = v
        elif kk.endswith(".lora_B.weight"): mods.setdefault(kk[:-14], {})["B"] = v
    P = dict(model.named_parameters()); n = 0
    for pf, d in mods.items():
        wk = pf + ".weight"
        if wk in P and "A" in d and "B" in d:
            with torch.no_grad():
                P[wk].add_(((d["B"].float() @ d["A"].float()) * sc).to(P[wk].device, P[wk].dtype)); n += 1
    print(f"merged adapter {n} modules (r={_r:g} alpha={_a:g} scale={sc:g})", flush=True)
try: model.disable_talker()
except Exception: pass
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok = proc.tokenizer

YES = sorted({i for s in ("yes", "Yes", " yes", " Yes") for i in tok.encode(s, add_special_tokens=False)[:1]})
NO = sorted({i for s in ("no", "No", " no", " No") for i in tok.encode(s, add_special_tokens=False)[:1]})


def prefix_upto_answer(row):
    sm = next(m["content"] for m in row["prompt"] if m["role"] == "system")
    q = next(m["content"] for m in row["prompt"] if m["role"] == "user").replace("<video>", "").replace("<audio>", "").strip()
    msgs = [{"role": "system", "content": [{"type": "text", "text": sm}]},
            {"role": "user", "content": [{"type": "video", "video": row["videos"][0], "nframes": NFRAMES},
                                         {"type": "audio", "audio": row["audios"][0]},
                                         {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    text += row["response"].split("<answer>")[0] + "<answer>"
    audios, images, videos, vkw = process_mm_info(msgs, use_audio_in_video=False, return_video_kwargs=True)
    if isinstance(vkw.get("fps"), (list, tuple)) and len(vkw["fps"]): vkw = {**vkw, "fps": vkw["fps"][0]}
    return proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                padding=True, use_audio_in_video=False, **vkw)


recs, t0 = [], time.time()
for i in range(len(df)):
    row = df.iloc[i]
    try:
        inp = prefix_upto_answer(row).to("cuda")
        with torch.no_grad():
            logits = model.thinker(**inp).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
    except Exception as e:
        print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
        continue
    ei = row["extra_info"]; gt = str(row["reward_model"]["ground_truth"]).lower()
    z = py + pn
    p_gt = (py if gt == "yes" else pn) / z if z > 0 else 0.5
    recs.append({"i": int(i), "group_id": ei["group_id"], "task": ei["task"],
                 "content_swapped": ei["content_swapped"], "gt": gt,
                 "p_yes": py, "p_no": pn, "p_gt": p_gt, "p_top": max(py, pn) / z if z > 0 else 0.5,
                 "pred": "yes" if py >= pn else "no"})
    if i % 500 == 0:
        print(f"  [{i}/{len(df)}] {(time.time()-t0)/60:.0f}m elapsed", flush=True)

json.dump(recs, open(OUT, "w"))
print(f"wrote {OUT}  n={len(recs)}", flush=True)
