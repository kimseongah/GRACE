import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grace.prompts import STRUCTURED_SYS

QA = os.environ.get("AF_QA", "data/AVHBench/avhbench_qa.json")
MEDIA = os.environ.get("AF_MEDIA", "data/AVHBench")
OUT = os.environ.get("AF_OUT", "data/avhbench_eval.parquet")

TASK = {"Audio-driven Video Hallucination": "ADVH",
        "Video-driven Audio Hallucination": "VDAH",
        "AV Matching": "AVM"}

have = ({os.path.splitext(f)[0] for f in os.listdir(f"{MEDIA}/videos")}
        & {os.path.splitext(f)[0] for f in os.listdir(f"{MEDIA}/audios")})
qa = json.load(open(QA))
rows, stats, missing = [], collections.Counter(), 0
for r in qa:
    if r["task"] not in TASK: continue
    if str(r["label"]).lower() not in ("yes", "no"): continue
    vid = r["video_id"]
    if vid not in have: missing += 1; continue
    t = TASK[r["task"]]; stats[t] += 1
    rows.append({
        "data_source": f"avhbench/{t}",
        "prompt": [{"role": "system", "content": STRUCTURED_SYS},
                   {"role": "user", "content": "<video>\n<audio>\n" + r["text"]}],
        "videos": np.array([f"{MEDIA}/videos/{vid}.mp4"]),
        "audios": np.array([f"{MEDIA}/audios/{vid}.wav"]),
        "reward_model": {"style": "rule", "ground_truth": str(r["label"]).lower()},
        "extra_info": {"task": t, "is_counterfactual": False, "ablation": "none", "video_id": vid},
    })
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"wrote {OUT}  rows={len(df)}  per-task={dict(stats)}  (media-missing skipped: {missing})")
