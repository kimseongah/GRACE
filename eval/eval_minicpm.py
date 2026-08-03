import os,json,re,time,numpy as np,pandas as pd,torch
from PIL import Image
from decord import VideoReader, cpu
import librosa
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict, TaskType
from safetensors.torch import load_file
import signal
from contextlib import contextmanager
P=os.environ.get("CE_MODEL","openbmb/MiniCPM-o-2_6")
EVAL=os.environ["CE_EVAL"]; MROOT=os.environ.get("CE_MEDIA_ROOT","data/AVHBench")
OUT=os.environ["CE_OUT"]; RAW=os.environ.get("CE_RAW",OUT.replace(".json","_raw.jsonl"))
LIMIT=int(os.environ.get("CE_LIMIT","0")); NF=int(os.environ.get("VGG_NFRAMES","8"))
ECE_BINS=int(os.environ.get("CE_ECE_BINS","15"))
ADAPTER=os.environ.get("CE_ADAPTER",""); ADAPTER2=os.environ.get("CE_ADAPTER2","")
NATIVE=os.environ.get("CE_NATIVE","")=="1"; PROBE=os.environ.get("CE_PROBE","")=="1"
@contextmanager
def tl(s):
    def h(a,b): raise TimeoutError()
    o=signal.signal(signal.SIGALRM,h); signal.alarm(s)
    try: yield
    finally: signal.alarm(0); signal.signal(signal.SIGALRM,o)
def remap(p):
    for mk in ("/AVHBench/","/CMM/","/VGGSound/"):
        i=p.find(mk)
        if i>=0: return MROOT.rstrip("/")+p[i+len(mk)-1:]
    return p
def read_video_frames(path, nf):
    try:
        vr=VideoReader(path,ctx=cpu(0)); n=len(vr)
        if n<=0: raise RuntimeError("empty")
        idx=np.linspace(0,n-1,nf).astype(int)
        return [Image.fromarray(vr[j].asnumpy()) for j in idx]
    except Exception:
        import av
        ct=av.open(path); frames=[fr.to_image() for fr in ct.decode(video=0)]; ct.close()
        n=len(frames)
        if n<=0: raise RuntimeError("empty video (both readers)")
        idx=np.linspace(0,n-1,nf).astype(int)
        return [frames[j] for j in idx]
print("load MiniCPM-o...",flush=True)
model=AutoModel.from_pretrained(P,trust_remote_code=True,attn_implementation="sdpa",torch_dtype=torch.bfloat16,init_vision=True,init_audio=True,init_tts=False).eval().cuda()
tok=AutoTokenizer.from_pretrained(P,trust_remote_code=True)
if ADAPTER:
    cfg=json.load(open(os.path.join(ADAPTER,"adapter_config.json")))
    lc=LoraConfig(r=cfg.get("r",32),lora_alpha=cfg.get("lora_alpha",16),target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],task_type=TaskType.CAUSAL_LM,lora_dropout=0.0)
    model.llm=inject_adapter_in_model(lc,model.llm)
    sd=load_file(os.path.join(ADAPTER,"adapter_model.safetensors"))
    set_peft_model_state_dict(model.llm,sd)
    print(f"[lora] loaded grounding {len(sd)} tensors from {ADAPTER}",flush=True)
    if ADAPTER2:
        sd2=load_file(os.path.join(ADAPTER2,"adapter_model.safetensors"))
        import torch as _t
        cfg2=json.load(open(os.path.join(ADAPTER2,"adapter_config.json"))); sc2=cfg2.get("lora_alpha",16)/cfg2.get("r",32)
        mods={}
        for k,v in sd2.items():
            kk=k.replace("base_model.model.","")
            if kk.endswith(".lora_A.weight"): mods.setdefault(kk[:-14],{})["A"]=v
            elif kk.endswith(".lora_B.weight"): mods.setdefault(kk[:-14],{})["B"]=v
        Pm=dict(model.llm.named_parameters()); n=0
        for pf,d in mods.items():
            wk=pf+".base_layer.weight" if pf+".base_layer.weight" in Pm else pf+".weight"
            if wk in Pm and "A" in d and "B" in d:
                with _t.no_grad(): Pm[wk].add_(((d["B"].float()@d["A"].float())*sc2).to(Pm[wk].device,Pm[wk].dtype)); n+=1
        print(f"[lora] merged conf {n} modules from {ADAPTER2}",flush=True)
_A=re.compile(r"<answer>\s*(yes|no)\s*</answer>",re.I); _C=re.compile(r"<conf>\s*([01](?:\.\d+)?|\.\d+)\s*</conf>",re.I)
_BARE=re.compile(r"\b(yes|no)\b",re.I)
def parse(t):
    if not t: return None,0.5,False
    m=_A.search(t); ans=m.group(1).lower() if m else None
    if ans is None:
        b=_BARE.search(t)
        if b: ans=b.group(1).lower()
    c=_C.search(t); return ans,(float(c.group(1)) if c else 0.5),bool(c)
NATIVE_SYS="Answer the question with a single word: yes or no."
df=pd.read_parquet(EVAL); df["task"]=df["extra_info"].apply(lambda x:x.get("task"))
df=df[df["task"].isin(["ADVH","VDAH","AVM","CMM"])].reset_index(drop=True)
if LIMIT: df=df.head(LIMIT).reset_index(drop=True)
print(f"[data] n={len(df)} adapter={ADAPTER or '-'} adapter2={ADAPTER2 or '-'} native={NATIVE}",flush=True)
rf=open(RAW,"w"); recs=[]; t0=time.time()
for i,row in df.iterrows():
    gt=str(row["reward_model"]["ground_truth"]).lower()
    sm=next(m["content"] for m in row["prompt"] if m["role"]=="system") if not NATIVE else NATIVE_SYS
    q=next(m["content"] for m in row["prompt"] if m["role"]=="user").replace("<video>","").replace("<audio>","").strip()
    vs=list(row["videos"]) if row["videos"] is not None else []; aus=list(row["audios"]) if row["audios"] is not None else []
    resp=""
    try:
        with tl(90):
            content=[]
            if vs:
                content+=read_video_frames(remap(vs[0]),NF)
            if aus:
                a,_=librosa.load(remap(aus[0]),sr=16000); content.append(a)
            content.append(q)
            resp=model.chat(msgs=[{"role":"system","content":sm},{"role":"user","content":content}],tokenizer=tok,
                            omni_input=bool(vs and aus),sampling=False,max_new_tokens=128,use_tts_template=False,generate_audio=False,
                            max_slice_nums=1,use_image_id=False)
    except Exception as e: print(f"[skip {i}] {type(e).__name__}",flush=True)
    rr=resp if isinstance(resp,str) else str(resp)
    ans,conf,hc=parse(rr)
    rec=dict(task=row["task"],gt=gt,resp=rr[:120],ans=ans,conf=float(conf),has_conf=hc)
    recs.append(rec); rf.write(json.dumps(rec,ensure_ascii=False)+"\n"); rf.flush()
    if i%200==0: print(f"[{i+1}/{len(df)}] {(time.time()-t0)/60:.0f}m elapsed",flush=True)
rf.close()
def metrics(rs):
    comm=[r for r in rs if r["ans"] in ("yes","no")]
    acc=sum(1 for r in comm if r["ans"]==r["gt"])/len(comm) if comm else 0
    allc=sum(1 for r in rs if r["ans"]==r["gt"])/len(rs)
    cf=np.array([r["conf"] for r in comm]); co=np.array([int(r["ans"]==r["gt"]) for r in comm]) if comm else np.array([])
    ece=0.0
    if len(comm):
        idx=np.clip((cf*ECE_BINS).astype(int),0,ECE_BINS-1)
        for b in range(ECE_BINS):
            m=idx==b
            if m.sum(): ece+=m.mean()*abs(co[m].mean()-cf[m].mean())
    def auroc(s,y):
        s=np.array(s);y=np.array(y);pos=s[y==1];neg=s[y==0]
        if len(pos)==0 or len(neg)==0: return None
        allv=np.concatenate([pos,neg]);o=allv.argsort();r=np.empty(len(allv));r[o]=np.arange(1,len(allv)+1)
        return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
    au=auroc(cf,co) if len(comm) else None
    return dict(n=len(rs),n_committed=len(comm),acc=round(allc,4),
                has_conf_rate=round(sum(r["has_conf"] for r in rs)/len(rs),4),expr_auroc=round(au,4) if au else None,
                ece=round(ece,4))
res={"overall":metrics(recs),"per_task":{t:metrics([r for r in recs if r["task"]==t]) for t in set(r["task"] for r in recs)}}
json.dump(res,open(OUT,"w"),indent=1); print("RESULT",json.dumps(res["overall"]),flush=True)
