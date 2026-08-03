import os,json,time,random,numpy as np,pandas as pd,torch
from PIL import Image
from decord import VideoReader, cpu
import librosa
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, inject_adapter_in_model, TaskType, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file, load_file
P=os.environ.get("MC_MODEL","openbmb/MiniCPM-o-2_6")
DATA=os.environ["WS_DATA"]; OUT=os.environ["WS_OUT"]
LR=float(os.environ.get("WS_LR","5e-5"))
EPOCHS=int(os.environ.get("WS_EPOCHS","1")); ACCUM=int(os.environ.get("WS_ACCUM","8"))
NF=int(os.environ.get("VGG_NFRAMES","8")); SEED=int(os.environ.get("WS_SEED","0"))
R=int(os.environ.get("WS_R","32")); ALPHA=int(os.environ.get("WS_ALPHA","16"))
MROOT=os.environ.get("WS_MEDIA_ROOT","")
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
print("load MiniCPM-o...",flush=True)
model=AutoModel.from_pretrained(P,trust_remote_code=True,attn_implementation="sdpa",torch_dtype=torch.bfloat16,init_vision=True,init_audio=True,init_tts=False).cuda()
model.config.use_cache=False
tok=AutoTokenizer.from_pretrained(P,trust_remote_code=True); proc=AutoProcessor.from_pretrained(P,trust_remote_code=True)
lora=LoraConfig(r=R,lora_alpha=ALPHA,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],task_type=TaskType.CAUSAL_LM,lora_dropout=0.0)
model.llm=inject_adapter_in_model(lora,model.llm)
WS_INIT=os.environ.get("WS_INIT","")
if WS_INIT:
    set_peft_model_state_dict(model.llm,load_file(os.path.join(WS_INIT,"adapter_model.safetensors")))
    print(f"[WS_INIT] loaded grounding adapter from {WS_INIT}",flush=True)
for n,pp in model.llm.named_parameters(): pp.requires_grad=("lora_" in n)
model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads()
for m in [getattr(model,"vpm",None),getattr(model,"apm",None),getattr(model,"resampler",None)]:
    if m is not None:
        for p in m.parameters(): p.requires_grad=False
ntr=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad); print(f"trainable={ntr}",flush=True)
def remap(p):
    if not MROOT: return p
    for mk in ("/VGGSound/","/AVHBench/","/CMM/"):
        i=p.find(mk)
        if i>=0: return MROOT.rstrip("/")+p[i+len(mk)-1:]
    return p
df=pd.read_parquet(DATA)
df=df.sample(frac=1,random_state=SEED).reset_index(drop=True)
print(f"data={DATA} n={len(df)} lr={LR} epochs={EPOCHS} accum={ACCUM}",flush=True)
def build(row):
    sysm=next(m["content"] for m in row["prompt"] if m["role"]=="system")
    q=next(m["content"] for m in row["prompt"] if m["role"]=="user").replace("<video>","").replace("<audio>","").strip()
    images=[]; audios=[]; audio_parts=[]; cur=[]
    vs=list(row["videos"]) if row["videos"] is not None else []; aus=list(row["audios"]) if row["audios"] is not None else []
    if vs:
        vr=VideoReader(remap(vs[0]),ctx=cpu(0)); idx=np.linspace(0,len(vr)-1,NF).astype(int)
        for j in idx: images.append(Image.fromarray(vr[j].asnumpy())); cur.append("(<image>./</image>)")
    if aus:
        a,_=librosa.load(remap(aus[0]),sr=16000); audios.append(a); audio_parts.append(1); cur.append("(<audio>./</audio>)")
    cur.append(q)
    msgs=[{"role":"system","content":sysm},{"role":"user","content":"".join(cur)}]
    ptext=proc.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,chat_template=model.default_tts_chat_template)
    resp=row["response"]+tok.eos_token
    kw=dict(max_slice_nums=1,use_image_id=False,return_tensors="pt",max_length=8192)
    p_in=proc([ptext],[images],[audios],[audio_parts],**kw); Lp=p_in["input_ids"][0].shape[-1]
    f_in=proc([ptext+resp],[images],[audios],[audio_parts],**kw); Lf=f_in["input_ids"][0].shape[-1]
    labels=f_in["input_ids"].clone().long(); labels[0,:Lp]=-100
    f_in["position_ids"]=torch.arange(Lf).unsqueeze(0)
    return f_in.to("cuda"), labels.to("cuda")
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
def save_adapter(outdir):
    os.makedirs(outdir,exist_ok=True)
    _sd=get_peft_model_state_dict(model.llm); _sd={k:v.to(torch.bfloat16).contiguous() for k,v in _sd.items()}
    save_file(_sd,os.path.join(outdir,"adapter_model.safetensors"))
    _c=lora.to_dict(); _c["target_modules"]=list(_c["target_modules"]) if not isinstance(_c["target_modules"],list) else _c["target_modules"]
    json.dump({k:(str(v) if not isinstance(v,(list,dict,str,int,float,bool,type(None))) else v) for k,v in _c.items()},open(os.path.join(outdir,"adapter_config.json"),"w"),indent=1)
model.train(); step=0; al=0.0; nsk=0; t0=time.time(); opt.zero_grad()
tot=EPOCHS*len(df)
for ep in range(EPOCHS):
    for i in range(len(df)):
        try:
            data,labels=build(df.iloc[i])
            loss=model(data,labels=labels).loss/ACCUM
            loss.backward(); al+=float(loss)
        except Exception as e:
            nsk+=1;
            if nsk<=5: print(f"[skip {i}] {type(e).__name__}: {str(e)[:60]}",flush=True); continue
            else: continue
        gi=ep*len(df)+i
        if (gi+1)%ACCUM==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad(); step+=1
            if step%10==0: print(f"ep{ep} it{i+1}/{len(df)} step{step} loss={al:.4f} ({time.time()-t0:.0f}s, {(time.time()-t0)/max(1,gi+1):.2f}s/it) skip={nsk}",flush=True); al=0
save_adapter(OUT)
print(f"[done] saved adapter to {OUT} steps={step}",flush=True)
