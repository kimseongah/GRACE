import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grace.prompts import STRUCTURED_SYS

ANNO = os.environ.get("CMM_ANNO", "data/CMM/cmm_annotations.jsonl")
MEDIA = os.environ.get("CMM_MEDIA", "data/CMM/reorg_raw_files")
OUT = os.environ.get("CMM_OUT", "data/cmm_eval.parquet")

def slots(sub_category, media, path):
    is_wav = media.lower().endswith(".wav")
    if sub_category == "audio-language" or is_wav:
        return np.array([]), np.array([path]), "<audio>\n"
    if sub_category == "visual-audio-language":
        return np.array([path]), np.array([path]), "<video>\n<audio>\n"
    return np.array([path]), np.array([]), "<video>\n"

rows, stats = [], collections.Counter()
for i, line in enumerate(open(ANNO)):
    a = json.loads(line)
    cat, sub, media = a["category"], a["sub_category"], a["media"]
    path = f"{MEDIA}/{cat}/{sub}/{media}"
    v, au, tag = slots(sub, media, path)
    rows.append({
        "data_source": f"cmm/{cat}",
        "prompt": [{"role": "system", "content": STRUCTURED_SYS},
                   {"role": "user", "content": tag + a["question"]}],
        "videos": v, "audios": au,
        "reward_model": {"style": "rule", "ground_truth": str(a["answer"]).lower()},
        "extra_info": {"task": "CMM", "item_id": a.get("item_id", i), "category": cat,
                       "sub_category": sub, "modality": a.get("modality"),
                       "granularity": a.get("granularity"), "correlation_type": a.get("correlation_type")},
    })
    stats[cat] += 1
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"wrote {OUT}  rows={len(df)}  per-category={dict(stats)}")
