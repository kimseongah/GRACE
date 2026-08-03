import os, sys, json, random
import pandas as pd
CT_TRAIN = os.environ.get("CT_TRAIN", "data/vggsound_train.parquet")
CT_PROBE = os.environ.get("CT_PROBE", "data/qwen_probe.json")
CT_OUT = os.environ.get("CT_OUT", "data/vggsound_conftarget.parquet")
rng = random.Random(0)
THINK = ["Checking the queried event against the available audio and video evidence.",
         "Assessing whether the audio and video support the question.",
         "Looking for the queried event across the audio and video streams."]
ABL2M = {"mute_audio": "audio", "blank_video": "video"}

df = pd.read_parquet(CT_TRAIN)
probe = json.load(open(CT_PROBE))
recs = probe["records"] if isinstance(probe, dict) else probe
by_i = {int(r["i"]): r for r in recs if "i" in r}
if not by_i:
    assert len(recs) == len(df), f"len mismatch probe {len(recs)} vs train {len(df)}"
    by_i = dict(enumerate(recs))
print(f"train={len(df)} probed={len(by_i)}")
df["gt"] = df["reward_model"].apply(lambda x: str(x["ground_truth"]).lower())
df["abl"] = df["extra_info"].apply(lambda x: x.get("ablation"))
df["cf"] = df["extra_info"].apply(lambda x: bool(x.get("is_counterfactual")))

rows = []
for i, r in df.iterrows():
    rec = by_i.get(int(i))
    if rec is None:
        continue
    py, pn, pu = float(rec["p_yes"]), float(rec["p_no"]), float(rec.get("p_unsure", 0.5))
    th = THINK[rng.randrange(len(THINK))]
    if r["cf"]:
        m = ABL2M.get(r["abl"]); conf = pu
        if m is None: continue
        resp = f"<think> {th} </think>\n<answer>unsure</answer>\n<missing>{m}</missing>\n<conf>{conf:.2f}</conf>"
    else:
        gt = r["gt"]; conf = py if gt == "yes" else pn
        resp = f"<think> {th} </think>\n<answer>{gt}</answer>\n<conf>{conf:.2f}</conf>"
    rows.append({"data_source": r["data_source"], "prompt": r["prompt"], "videos": r["videos"],
                 "audios": r["audios"], "extra_info": r["extra_info"], "response": resp})
out = pd.DataFrame(rows)
out.to_parquet(CT_OUT, index=False)
print(f"wrote {CT_OUT}  n={len(out)}")
