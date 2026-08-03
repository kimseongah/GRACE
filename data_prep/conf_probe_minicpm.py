import os,json,re,time,numpy as np,pandas as pd,torch
from PIL import Image
from decord import VideoReader, cpu
import librosa, signal
from contextlib import contextmanager
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict, TaskType
from safetensors.torch import load_file
P=os.environ.get("PC_MODEL","openbmb/MiniCPM-o-2_6")
DATA=os.environ["WS_DATA"]; ADAPTER=os.environ["PC_ADAPTER"]; OUT=os.environ["OUT_PARQUET"]
NF=8
NATIVE="Answer the question with a single word: yes or no."; BOIL="Checking the queried event against the available audio and video evidence."
torch.manual_seed(0)
@contextmanager
def tl(s):
    def h(a,b): raise TimeoutError()
    o=signal.signal(signal.SIGALRM,h); signal.alarm(s)
    try: yield
    finally: signal.alarm(0); signal.signal(signal.SIGALRM,o)
print("load MiniCPM + grounding LoRA...",flush=True)
model=AutoModel.from_pretrained(P,trust_remote_code=True,attn_implementation="sdpa",torch_dtype=torch.bfloat16,init_vision=True,init_audio=True,init_tts=False).eval().cuda()
tok=AutoTokenizer.from_pretrained(P,trust_remote_code=True)
_cfgp=os.path.join(ADAPTER,"adapter_config.json")
_cfg=json.load(open(_cfgp)) if os.path.exists(_cfgp) else {}
lc=LoraConfig(r=int(_cfg.get("r",32)),lora_alpha=int(_cfg.get("lora_alpha",16)),target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],task_type=TaskType.CAUSAL_LM,lora_dropout=0.0)
model.llm=inject_adapter_in_model(lc,model.llm); set_peft_model_state_dict(model.llm,load_file(os.path.join(ADAPTER,"adapter_model.safetensors")))
print("grounding LoRA loaded",flush=True)
_cap={}
def _decode_p(inputs_embeds, tokenizer, attention_mask, **kwargs):
    kwargs.pop("output_hidden_states",None); kwargs.pop("return_dict_in_generate",None)
    term=[tokenizer.convert_tokens_to_ids(i) for i in model.terminators]
    o=model.llm.generate(inputs_embeds=inputs_embeds,pad_token_id=0,eos_token_id=term,attention_mask=attention_mask,output_hidden_states=True,return_dict_in_generate=True,output_scores=True,**kwargs)
    _cap["scores"]=o.scores; return o
model._decode=_decode_p
def _ids(ws):
    s=set()
    for w in ws:
        e=tok.encode(w,add_special_tokens=False)
        if e: s.add(e[0])
    return sorted(s)
YES,NO=_ids(["yes","Yes","YES"]),_ids(["no","No","NO"])
df=pd.read_parquet(DATA).reset_index(drop=True)
print(f"probe n={len(df)}",flush=True)
rows=[]; t0=time.time()
for i in range(len(df)):
    row=df.iloc[i]; _ma=re.search(r"<answer>\s*(yes|no)\s*</answer>",str(row["response"]),re.I); gt=(_ma.group(1).lower() if _ma else "")
    q=next(m["content"] for m in row["prompt"] if m["role"]=="user").replace("<video>","").replace("<audio>","").strip()
    vs=list(row["videos"]) if row["videos"] is not None else []; aus=list(row["audios"]) if row["audios"] is not None else []
    conf=0.5; _cap.clear()
    try:
        with tl(60):
            content=[]
            if vs:
                vr=VideoReader(vs[0],ctx=cpu(0)); idx=np.linspace(0,len(vr)-1,NF).astype(int); content+=[Image.fromarray(vr[j].asnumpy()) for j in idx]
            if aus:
                a,_=librosa.load(aus[0],sr=16000); content.append(a)
            content.append(q)
            model.chat(msgs=[{"role":"system","content":NATIVE},{"role":"user","content":content}],tokenizer=tok,omni_input=bool(vs and aus),sampling=False,max_new_tokens=4,use_tts_template=False,generate_audio=False,max_slice_nums=1,use_image_id=False)
            if _cap.get("scores"):
                p=torch.softmax(_cap["scores"][0][0].float(),-1); py=float(p[YES].sum()); pn=float(p[NO].sum()); z=py+pn+1e-9
                conf=(py/z) if gt=="yes" else ((pn/z) if gt=="no" else 0.5)
    except Exception as e:
        if i<3: print("skip",i,type(e).__name__,flush=True)
    resp=f"<think> {BOIL} </think>\n<answer>{gt}</answer>\n<conf>{conf:.2f}</conf>"
    d=row.to_dict(); d["response"]=resp; rows.append(d)
    if i%500==0: print(f"[{i+1}/{len(df)}] {(time.time()-t0)/60:.0f}m elapsed",flush=True)
out=pd.DataFrame(rows); out.to_parquet(OUT)
print(f"wrote {OUT} n={len(out)}",flush=True)
