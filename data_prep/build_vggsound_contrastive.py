import os, sys, json, csv, random, collections, glob
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_HERE, ".."))
from grace.prompts import STRUCTURED_SYS as SYSTEM_PROMPT

CSV = os.environ.get("VGG_CSV", "data/VGGSound/vggsound.csv")
MAP = os.environ.get("VGG_MAP", "data/VGGSound/vgg_label_qa_map.json")
VDIR = os.environ.get("VGG_VIDEODIR", "data/VGGSound/video")
ADIR = os.environ.get("VGG_AUDIODIR", "data/VGGSound/audio")
OUT = os.environ.get("VGG_OUT", "data/vggsound_train.parquet")
SEED = int(os.environ.get("VGG_SEED", "0")); rng = random.Random(SEED)

THINK = ["Assessing whether the audio and video support the question.",
         "Checking the queried event against the available audio and video evidence.",
         "Looking for the queried event across the audio and video streams."]
CONF = "0.7"
DECIDING = {"VDAH": "audio", "ADVH": "video"}

def q_of(task, subj): return f"Is {subj} making sound in the audio?" if task == "VDAH" else f"Is {subj} visible in the video?"

def the(np_phrase):
    p = (np_phrase or "").strip()
    if not p: return p
    low = p.lower()
    for art in ("a ", "an ", "the "):
        if low.startswith(art): return "the " + p[len(art):]
    return p

def audio_subj(m): return the(m.get("visual_subject") or "")
def video_subj(m): return the(m["visual_subject"]) if (m.get("visual_reliable") and m.get("visual_subject")) else None
def stem_of(ytid, start): return f"{ytid}_{int(start):06d}"

def main():
    lm = {d["label"]: d for d in json.load(open(MAP))}
    label_of = {}
    for r in csv.reader(open(CSV)):
        if len(r) >= 4: label_of[stem_of(r[0], r[1])] = r[2]

    present = {}
    for f in glob.glob(os.path.join(VDIR, "*.mp4")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in label_of and label_of[stem] in lm and os.path.exists(os.path.join(ADIR, stem + ".wav")):
            present[stem] = label_of[stem]
    print(f"present clips (label+wav+mapped): {len(present)}", flush=True)
    if not present:
        print("NO present clips -- check VDIR/ADIR/naming"); return

    by_cat = collections.defaultdict(list)
    for stem, lab in present.items(): by_cat[lm[lab]["category"]].append(stem)
    cats = list(by_cat)

    def donor(cat, task):
        for _ in range(40):
            oc = rng.choice([c for c in cats if c != cat])
            ds = rng.choice(by_cat[oc]); dm = lm[label_of[ds]]
            if task == "ADVH" and not video_subj(dm): continue
            if task == "VDAH" and not audio_subj(dm): continue
            return ds, dm
        return None, None

    vp = lambda s: os.path.join(VDIR, s + ".mp4")
    ap = lambda s: os.path.join(ADIR, s + ".wav")

    def row(gid, task, q, vstem, astem, gt, swapped, orig, i, subj):
        return {
            "data_source": f"vggsound/{task}",
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": "<video>\n<audio>\n" + q}],
            "videos": np.array([vp(vstem)]), "audios": np.array([ap(astem)]),
            "reward_model": {"style": "rule", "ground_truth": gt},
            "response": f"<think> {THINK[i % len(THINK)]} </think>\n<answer>{gt}</answer>\n<conf>{CONF}</conf>",
            "extra_info": {"group_id": gid, "task": task, "is_counterfactual": False, "ablation": "none",
                           "evidence_weak": False, "video_id": orig, "question": q,
                           "content_swapped": swapped, "donor_video_id": (vstem if swapped == "video" else astem) if swapped != "none" else None,
                           "orig_gt": gt, "answer": gt, "subject": subj, "source": "vggsound"},
        }

    rows, i = [], 0
    for stem in sorted(present):
        m = lm[present[stem]]; cat = m["category"]
        for task in ("VDAH", "ADVH"):
            S = audio_subj(m) if task == "VDAH" else video_subj(m)
            if not S: continue
            ds, dm = donor(cat, task)
            if not ds: continue
            X = audio_subj(dm) if task == "VDAH" else video_subj(dm)
            sv, sa = (ds, stem) if task == "ADVH" else (stem, ds)
            gA = f"vg_{stem}_{task}_A_{i:06d}"
            rows.append(row(gA, task, q_of(task, S), stem, stem, "yes", "none", stem, i, S)); i += 1
            rows.append(row(gA, task, q_of(task, S), sv, sa, "no", DECIDING[task], stem, i, S)); i += 1
            gB = f"vg_{stem}_{task}_B_{i:06d}"
            rows.append(row(gB, task, q_of(task, X), stem, stem, "no", "none", stem, i, X)); i += 1
            rows.append(row(gB, task, q_of(task, X), sv, sa, "yes", DECIDING[task], stem, i, X)); i += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} rows={len(df)} groups={df['extra_info'].apply(lambda x: x['group_id']).nunique()}", flush=True)

if __name__ == "__main__":
    main()
