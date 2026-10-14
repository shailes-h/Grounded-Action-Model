#!/usr/bin/env python
"""Click annotator for yam episodes — two marks only: pick and place. No breakpoints.

    python labeling/annotator.py --config configs/my_task.yaml [--port 8870]
    -> open http://127.0.0.1:8870  (forward the port if remote)

Interaction: pick an episode, scrub frames (slider / a,d), press 1 (pick) or 2 (place)
to choose the mark, click on the image to drop it — clicking near an existing mark
deletes it. Everything autosaves to `<cache-sibling> annotations/<episode>.json`:

    {"points": [{"role": "pick"|"place", "frame": t, "point": [x, y]}, ...]}

Feed the result back through the config:

    data:
      clicks: {source: annotations, annotations_dir: /path/annotations}

which builds each episode's clicks as [all pick marks (click order), then all place
marks], each held constant over time — order must match `data.slots` (manipulated
object first).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402

app = Flask(__name__)
CFG = None
ANN = None
_open = {"path": None, "ep": None}


def _ep(path):
    if _open["path"] != path:
        if _open["ep"] is not None:
            _open["ep"].close()
        _open["ep"] = load_episode(path, CFG)
        _open["path"] = path
    return _open["ep"]


def _ann_path(name):
    return os.path.join(ANN, name + ".json")


@app.get("/api/episodes")
def api_episodes():
    out = []
    for p in list_episodes(CFG):
        n = os.path.splitext(os.path.basename(p))[0]
        done = os.path.exists(_ann_path(n))
        out.append({"name": n, "path": p, "done": done})
    slots = CFG["data"]["slots"]
    sequential = bool(CFG["data"]["clicks"].get("sequential", False))
    # how many of `slots` are picks vs places -- the loader concatenates
    # [picks..., places...] in that order, so this is the split point. No field for it
    # in the schema yet, so `data.clicks.n_picks` is optional; default splits evenly and
    # is very possibly wrong for your task -- set n_picks explicitly if picks != places.
    n_picks = CFG["data"]["clicks"].get("n_picks", len(slots) // 2)
    slot_labels = CFG["data"].get("slot_labels", slots)
    slot_choices = CFG["data"].get("slot_choices", [])
    return jsonify({"episodes": out, "slots": slots, "n_picks": n_picks,
                     "sequential": sequential, "slot_labels": slot_labels,
                     "slot_choices": slot_choices})


@app.get("/api/meta")
def api_meta():
    ep = _ep(request.args["path"])
    return jsonify({"T": int(ep.T)})


@app.get("/frame")
def frame():
    ep = _ep(request.args["path"])
    t = int(np.clip(int(request.args.get("t", 0)), 0, ep.T - 1))
    rgb = ep.rgb(t)
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.get("/api/ann")
def get_ann():
    p = _ann_path(request.args["name"])
    return jsonify(json.load(open(p)) if os.path.exists(p) else {"points": []})


@app.post("/api/ann")
def set_ann():
    d = request.get_json(force=True)
    os.makedirs(ANN, exist_ok=True)
    json.dump({"points": d.get("points", [])}, open(_ann_path(d["name"]), "w"), indent=1)
    return jsonify({"ok": True})


PAGE = """<!doctype html><meta charset=utf-8><title>yam annotator</title>
<style>
 body{background:#14161a;color:#dde;font:14px system-ui;margin:0;display:flex}
 #side{width:230px;min-width:230px;height:100vh;overflow:auto;border-right:1px solid #333;padding:8px}
 .row{padding:4px 6px;border-radius:5px;cursor:pointer}.row:hover{background:#242832}
 .row.cur{background:#2c3550}.done{color:#7c7}
 #main{padding:10px;flex:1}
 canvas{border:1px solid #333;max-width:100%}
 button{background:#2a2f3a;color:#dde;border:1px solid #444;border-radius:6px;padding:5px 12px;margin:2px;cursor:pointer}
 button.on{background:#1d4d2a;border-color:#3a7}
 button.on2{background:#6b4a17;border-color:#c93}
 button.on3{background:#2a4a6b;border-color:#39c}
 button.on4{background:#6b2a4a;border-color:#c39}
 input[type=range]{width:640px;vertical-align:middle}
 #hint{color:#889;margin-top:6px}
</style>
<div id=side></div>
<div id=main>
 <div id=toolbar></div>
 <canvas id=cv width=960 height=720></canvas>
 <div id=hint></div>
</div>
<script>
let eps=[],cur=-1,T=1,t=0,role='pick',pts=[],img=new Image(),W=0,H=0;
let SLOTS=[],NPICKS=0,SEQ=false,CHOICES=[];
let activeChoice='';  // sequential+choices: which choice button is active
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
// distinct colours per choice (pick=greens, place=oranges, extra=blues)
const CHOICE_COLORS={};
const PALETTE=['#3c5','#e93','#39f','#e3c','#5c3','#c93','#3cf','#c3e'];

async function boot(){
  const r=await (await fetch('api/episodes')).json();
  eps=r.episodes; SLOTS=r.slots; NPICKS=r.n_picks; SEQ=!!r.sequential;
  CHOICES=r.slot_choices||[];
  // assign a unique colour per choice
  CHOICES.forEach((c,i)=>{ CHOICE_COLORS[c]=PALETTE[i%PALETTE.length]; });
  buildToolbar();
  side(); if(eps.length) show(0);
}

function buildToolbar(){
  const tb=document.getElementById('toolbar');
  const hint=document.getElementById('hint');
  if(SEQ && CHOICES.length){
    // Choice mode: one button per action-object combo (e.g. "pick can", "place bowl")
    let html='';
    CHOICES.forEach((c,i)=>{
      html+=`<button id="bchoice${i}" onclick="setChoice('${c}',${i})">${i+1} · ${c}</button> `;
    });
    html+=`<span style="margin-left:18px">frame <input type=range id=sl min=0 max=0 value=0>
      <span id=fr>0</span>/<span id=tt>0</span></span>`;
    tb.innerHTML=html;
    hint.innerHTML='<b>Workflow:</b> select what to mark (1-'+CHOICES.length+
      ') → click on that object/location → it fills the next slot in sequence.<br>'+
      'Click a mark to delete it · a/d step frame · w/s ±10 · autosaves';
    setChoice(CHOICES[0],0);
  } else if(SEQ){
    // Sequential mode without choices (generic slots)
    let html='';
    SLOTS.forEach((s,i)=>{
      html+=`<button id="bslot${i}" onclick="setSlotDirect(${i})">${i+1} · ${s}</button> `;
    });
    html+=`<span style="margin-left:18px">frame <input type=range id=sl min=0 max=0 value=0>
      <span id=fr>0</span>/<span id=tt>0</span></span>`;
    tb.innerHTML=html;
    hint.textContent='1-'+SLOTS.length+' choose slot · click to place mark · click a mark to delete · a/d step frame · w/s ±10 · autosaves';
    setSlotDirect(0);
  } else {
    // Original pick/place mode
    tb.innerHTML=`<button id=bpick class=on onclick="setRole('pick')">1 · pick</button>
      <button id=bplace onclick="setRole('place')">2 · place</button>
      <span style="margin-left:18px">frame <input type=range id=sl min=0 max=0 value=0>
      <span id=fr>0</span>/<span id=tt>0</span></span>`;
    hint.textContent='1/2 choose the mark · click to drop it · click a mark to delete · a/d step frame · w/s ±10 · pick=green place=orange · autosaves';
  }
  document.getElementById('sl').oninput=function(){t=+this.value;draw();};
}

function setChoice(c,i){
  activeChoice=c;
  CHOICES.forEach((_,j)=>{
    const b=document.getElementById('bchoice'+j);
    if(b) b.className=(j===i)?(['on','on2','on3','on4'][j%4]):'';
  });
}
function setSlotDirect(i){
  activeChoice='__slot_'+i;
  SLOTS.forEach((_,j)=>{
    const b=document.getElementById('bslot'+j);
    if(b) b.className=(j===i)?(['on','on2','on3','on4'][j%4]):'';
  });
}

// --- original pick/place helpers (non-sequential) ---
function slotFor(roleName,n){
  const i=(roleName==='pick')?n:NPICKS+n;
  return (i<SLOTS.length)?SLOTS[i]:null;
}
function nextLabel(roleName){
  const n=pts.filter(p=>p.role===roleName).length;
  const s=slotFor(roleName,n);
  return s?`${roleName}: ${s}`:`${roleName} (extra, unused)`;
}

function updateButtons(){
  if(SEQ && CHOICES.length){
    // show which choices are used and which slot they got
    CHOICES.forEach((c,i)=>{
      const b=document.getElementById('bchoice'+i);
      if(!b) return;
      const used=pts.find(p=>p.label===c);
      b.textContent=(i+1)+' · '+c+(used?' → slot '+(pts.indexOf(used)+1)+' ✓':'');
    });
  } else if(SEQ){
    SLOTS.forEach((s,i)=>{
      const b=document.getElementById('bslot'+i);
      if(!b) return;
      const has=pts.some(p=>p.slot===i);
      b.textContent=(i+1)+' · '+s+(has?' ✓':'');
    });
  } else {
    document.getElementById('bpick').textContent='1 · '+nextLabel('pick');
    document.getElementById('bplace').textContent='2 · '+nextLabel('place');
  }
}

function side(){
  document.getElementById('side').innerHTML=eps.map((e,i)=>{
    let marker=e.done?'<span class=done>●</span> ':'○ ';
    return `<div class="row${i===cur?' cur':''}" onclick="show(${i})">${marker}${e.name}</div>`;
  }).join('');
}
async function show(i){
  cur=i; side();
  const e=eps[i];
  T=(await (await fetch('api/meta?path='+encodeURIComponent(e.path))).json()).T;
  const sl=document.getElementById('sl');
  sl.max=T-1; t=0;
  pts=(await (await fetch('api/ann?name='+e.name)).json()).points||[];
  draw(); updateButtons();
}
function draw(){
  const sl=document.getElementById('sl');
  sl.value=t; document.getElementById('fr').textContent=t; document.getElementById('tt').textContent=T-1;
  img.onload=()=>{W=img.width;H=img.height;cv.width=W;cv.height=H;cx.drawImage(img,0,0);marks();};
  img.src='frame?path='+encodeURIComponent(eps[cur].path)+'&t='+t+'&_='+Date.now();
}
function marks(){
  if(SEQ){
    // sequential mode: each point shows its label (object name) and slot position
    pts.forEach((p,i)=>{
      const col=CHOICE_COLORS[p.label]||PALETTE[i%PALETTE.length];
      cx.strokeStyle=cx.fillStyle=col;
      cx.lineWidth=2; cx.beginPath(); cx.arc(p.point[0],p.point[1],8,0,7); cx.stroke();
      cx.beginPath(); cx.arc(p.point[0],p.point[1],2.5,0,7); cx.fill();
      cx.font='bold 13px system-ui';
      const tag=(p.label||SLOTS[i]||'?')+' [slot '+(i+1)+']';
      // dark outline for readability
      cx.strokeStyle='#000'; cx.lineWidth=3; cx.strokeText(tag,p.point[0]+11,p.point[1]-8);
      cx.fillText(tag,p.point[0]+11,p.point[1]-8);
    });
  } else {
    const seen={pick:0,place:0};
    for(const p of pts){
      const s=slotFor(p.role,seen[p.role]); seen[p.role]++;
      cx.strokeStyle=cx.fillStyle=(p.role==='pick')?'#3c5':'#e93';
      cx.lineWidth=2; cx.beginPath(); cx.arc(p.point[0],p.point[1],8,0,7); cx.stroke();
      cx.beginPath(); cx.arc(p.point[0],p.point[1],2.5,0,7); cx.fill();
      cx.font='13px system-ui';
      cx.fillText((s||(p.role+' extra!'))+'@'+p.frame,p.point[0]+11,p.point[1]-8);
    }
  }
}
function setRole(r){role=r;
  document.getElementById('bpick').className=(r==='pick')?'on':'';
  document.getElementById('bplace').className=(r==='place')?'on2':'';}
cv.onclick=ev=>{
  const r=cv.getBoundingClientRect();
  const x=(ev.clientX-r.left)*cv.width/r.width, y=(ev.clientY-r.top)*cv.height/r.height;
  const hit=pts.findIndex(p=>Math.hypot(p.point[0]-x,p.point[1]-y)<12);
  if(hit>=0){
    pts.splice(hit,1);
  } else if(SEQ && CHOICES.length){
    // choice mode: check this choice isn't already used
    if(pts.some(p=>p.label===activeChoice)){
      // already placed — replace it
      const old=pts.findIndex(p=>p.label===activeChoice);
      pts.splice(old,1);
    }
    if(pts.length>=SLOTS.length){
      // all slots filled, don't add more
    } else {
      // append to the end — position = next slot
      pts.push({label:activeChoice,frame:t,
        point:[Math.round(x*10)/10,Math.round(y*10)/10]});
      // auto-select next unused choice
      for(let j=1;j<=CHOICES.length;j++){
        const next=CHOICES[(CHOICES.indexOf(activeChoice)+j)%CHOICES.length];
        if(!pts.some(p=>p.label===next)){
          setChoice(next,CHOICES.indexOf(next)); break;
        }
      }
    }
  } else if(SEQ){
    const si=parseInt(activeChoice.replace('__slot_',''));
    const old=pts.findIndex(p=>p.slot===si);
    if(old>=0) pts.splice(old,1);
    pts.push({slot:si,frame:t,point:[Math.round(x*10)/10,Math.round(y*10)/10]});
    for(let j=1;j<=SLOTS.length;j++){
      const next=(si+j)%SLOTS.length;
      if(!pts.some(p=>p.slot===next)){setSlotDirect(next);break;}
    }
  } else {
    pts.push({role:role,frame:t,point:[Math.round(x*10)/10,Math.round(y*10)/10]});
  }
  save(); draw(); updateButtons();
};
async function save(){
  await fetch('api/ann',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:eps[cur].name,points:pts})});
  eps[cur].done=pts.length>0; side();
}
document.onkeydown=e=>{
  if(SEQ && CHOICES.length){
    const n=parseInt(e.key);
    if(n>=1 && n<=CHOICES.length) setChoice(CHOICES[n-1],n-1);
  } else if(SEQ){
    const n=parseInt(e.key);
    if(n>=1 && n<=SLOTS.length) setSlotDirect(n-1);
  } else {
    if(e.key==='1')setRole('pick'); if(e.key==='2')setRole('place');
  }
  if(e.key==='a')t=Math.max(0,t-1); if(e.key==='d')t=Math.min(T-1,t+1);
  if(e.key==='w')t=Math.max(0,t-10); if(e.key==='s')t=Math.min(T-1,t+10);
  if('adws'.includes(e.key))draw();
};
boot();
</script>"""


@app.get("/")
def root():
    return PAGE


def main():
    global CFG, ANN
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="", help="annotations dir (default: annotations/ next to this file)")
    ap.add_argument("--port", type=int, default=8870)
    args = ap.parse_args()
    CFG = load_config(args.config)
    ANN = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotations")
    os.makedirs(ANN, exist_ok=True)
    print(f"[annotator] episodes={len(list_episodes(CFG))} annotations -> {ANN}")
    print(f"[annotator] open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
