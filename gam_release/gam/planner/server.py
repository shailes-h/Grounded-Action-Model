"""Molmo2 high-level planner service (language -> object points) with episode memory.

Role: language -> grounding, plus an episode memory. Molmo2 points at task-relevant
objects on frame 0; each point IS an object entity. The low-level tracker (e.g.
CoTracker in the agent) follows the pixels and reports them back, so the memory always knows
where every object is. Every time the planner looks at an image (/observe) it writes a
state note (what is held, where things are, what remains) and any task-relevant facts,
and later prompts carry that memory as context.

Runs as its own process (transformers>=4.57.1 for Molmo2; keep it separate from the
policy environment):
    python -m gam.planner.server                          # serve (MOLMO2_PORT, default 18092)
    python -m gam.planner.server --selftest img.png --instruction "Put the kettle on the tray"

HTTP API (JSON; all coordinates in PIXELS of the image you sent):
  stateless
    GET  /health            -> {"ok", "model", "loaded", "sessions"}
    POST /point             {"image_b64" | "image_raw_b64"+width+height[+bgr],
                             "queries":[{"name","text"}]} -> per-query points (raw + px)
    POST /plan              {"image_b64", "instruction", "want_place"?} -> {"pick","place",...}
  memory (per episode)
    POST /session           {"session_id","instruction","image_b64"[,"objects":preset]} -> frame-0 plan:
                            objects [{"id","name","role"(pick|place|other),"xy"}], state note.
                            Seed your tracker with these points, one per object.
    POST /observe           {"session_id","step","image_b64",
                             "tracked":[{"id","xy","visible"}],
                             "history_b64":[...],   (optional earlier frames, oldest first)
                             "events":{"releases":n}}  (gripper releases since last observe)
                            -> {"state","progress","step_index","reacquired":[{"id","xy"}],
                                "new_targets":[{"id","role","xy"}],"objects":[...]}
                            On frame 0 the planner also writes a sub-step PLAN; every observe
                            judges PROGRESS (which step is running) and, when the step changes,
                            points at the new pick/place targets (returned in new_targets —
                            add them to your tracker; the previous targets become role "done").
                            Coordinates are written into memory; Molmo2 reads the image
                            with the memory as context and appends a state note; objects
                            the tracker lost (visible=false) are re-pointed if Molmo2
                            can find them (feed them back into your tracker).
    GET  /memory?session_id=X   -> full memory dump (also persisted per session to
                                   MOLMO2_MEM_DIR/<session_id>.json on every write)

Point tag format (Molmo2 card): values are 0-1000 normalized; px = v/1000*W(H).
"""
import argparse
import base64
import io
import json
import os
import re
import threading
import time

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODEL_DIR = os.environ.get(
    "MOLMO2_MODEL", "allenai/Molmo2-8B")
PORT = int(os.environ.get("MOLMO2_PORT", "18092"))
MEM_DIR = os.environ.get("MOLMO2_MEM_DIR", "")          # persist memories here if set
MAX_NOTES_CTX = int(os.environ.get("MOLMO2_CTX_NOTES", "6"))   # recent notes in prompts

_model, _proc = None, None
_gen_lock = threading.Lock()
_sessions = {}
_sess_lock = threading.Lock()


# ----------------------------------------------------------------------------- model
def _load():
    global _model, _proc
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    print(f"[planner] loading {MODEL_DIR} ...", flush=True)
    _proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        os.environ.get("MOLMO2_DTYPE", "bf16")]           # weights on disk are fp32: ~17 GB in bf16
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=dtype, device_map="cuda:0")
    _model.eval()
    print(f"[planner] loaded in {time.time() - t0:.0f}s", flush=True)


def parse_points(txt, W, H):
    """Molmo2 point tags -> [(x_px, y_px)].

    Observed format (Molmo2-8B): <points coords="F i1 x1 y1 i2 x2 y2 ...">label</points>
    where F is the frame/image index, i_k a running point id and x/y are 0-1000
    normalized. Also accepts (id x y) triples without the frame prefix, plain (x y) pairs,
    and the legacy Molmo1 percent attributes x="63.5" y="42.1"."""
    pts = []
    for m in re.finditer(r'coords="([^"]+)"', txt):
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", m.group(1))]
        n = len(nums)
        if n >= 4 and (n - 1) % 3 == 0:                 # frame + (id x y)*
            pts += [(nums[i + 1], nums[i + 2]) for i in range(1, n, 3)]
        elif n >= 3 and n % 3 == 0:                     # (id x y)*
            pts += [(nums[i + 1], nums[i + 2]) for i in range(0, n, 3)]
        elif n >= 2 and n % 2 == 0:                     # (x y)*
            pts += list(zip(nums[0::2], nums[1::2]))
    if not pts:
        for x, y in re.findall(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', txt):
            pts.append((float(x) * 10.0, float(y) * 10.0))       # percent -> 0-1000
    return [(v[0] / 1000.0 * W, v[1] / 1000.0 * H) for v in pts]


def _ask(img, text, max_new_tokens=192):
    """img: PIL image or list of PIL images (consecutive frames, current one last)."""
    import torch
    _load()
    imgs = img if isinstance(img, (list, tuple)) else [img]
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]
                 + [{"type": "image", "image": im} for im in imgs]}]
    with _gen_lock:
        inputs = _proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(_model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        return _proc.tokenizer.decode(out[0, inputs["input_ids"].size(1):],
                                      skip_special_tokens=True)


def _decode_image(req):
    """image_b64 = encoded jpg/png; OR image_raw_b64 + width/height = raw RGB888 bytes
    (zero-dependency clients: color.tobytes())."""
    import numpy as np
    from PIL import Image
    if req.get("image_raw_b64"):
        w, h = int(req["width"]), int(req["height"])
        a = np.frombuffer(base64.b64decode(req["image_raw_b64"]), np.uint8).reshape(h, w, 3)
        if req.get("bgr"):
            a = a[:, :, ::-1]
        return Image.fromarray(a)
    img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
    return img.convert("RGB")


# ----------------------------------------------------------------------------- stateless
PICK_PROMPT = ('This is a robot manipulation scene. The robot must complete the task: '
               '"{instr}". Point to the single object the robot should pick up.')
PLACE_PROMPT = ('This is a robot manipulation scene. The robot must complete the task: '
                '"{instr}". Point to the exact location where the picked object should '
                'be placed (the target receptacle or spot).')
OTHER_PROMPT = ('This is a robot manipulation scene. The robot must complete the task: '
                '"{instr}". Besides the object to pick up and the place target, point to '
                'any other objects that matter for this task (obstacles in the way, '
                'reference objects mentioned in the task). If there are none, answer '
                '"none".')


def plan(img, instruction, want_place=True):
    W, H = img.size
    out = {"width": W, "height": H, "pick": None, "place": None, "raw": {}}
    raw = _ask(img, PICK_PROMPT.format(instr=instruction))
    out["raw"]["pick"] = raw
    p = parse_points(raw, W, H)
    if p:
        out["pick"] = [round(p[0][0], 1), round(p[0][1], 1)]
    if want_place:
        raw = _ask(img, PLACE_PROMPT.format(instr=instruction))
        out["raw"]["place"] = raw
        p = parse_points(raw, W, H)
        if p:
            out["place"] = [round(p[0][0], 1), round(p[0][1], 1)]
    return out


def point_queries(img, queries, max_new_tokens=192):
    W, H = img.size
    results = []
    for q in queries:
        raw = _ask(img, q["text"], max_new_tokens)
        results.append({"name": q.get("name", ""), "raw": raw,
                        "points": [[round(x, 1), round(y, 1)]
                                   for x, y in parse_points(raw, W, H)]})
    return {"width": W, "height": H, "results": results}


# ----------------------------------------------------------------------------- memory
class Memory:
    """Per-episode planner memory.
    objects: id -> {name, role, xy (latest px), visible, first_xy, history[(step,x,y,vis)]}
    notes:   [{step, state, facts}] one entry per image the planner looked at
    """

    def __init__(self, session_id, instruction, W, H):
        self.session_id = session_id
        self.instruction = instruction
        self.W, self.H = W, H
        self.objects = {}
        self.notes = []
        self.plan = []                 # sub-steps decided on frame 0 (strings)
        self.progress = ""             # latest "step k/N: ..." judgement
        self.created = time.time()
        self.last_step = 0

    # -- objects
    def add_object(self, name, role, xy, step=0):
        oid = f"o{len(self.objects)}"
        self.objects[oid] = {"id": oid, "name": name, "role": role,
                             "xy": [round(xy[0], 1), round(xy[1], 1)], "visible": True,
                             "first_xy": [round(xy[0], 1), round(xy[1], 1)],
                             "history": [(step, round(xy[0], 1), round(xy[1], 1), True)]}
        return oid

    def update_tracked(self, step, tracked):
        for t in tracked or []:
            o = self.objects.get(t.get("id"))
            if o is None or t.get("xy") is None:
                continue
            x, y = float(t["xy"][0]), float(t["xy"][1])
            vis = bool(t.get("visible", True))
            o["xy"], o["visible"] = [round(x, 1), round(y, 1)], vis
            o["history"].append((int(step), round(x, 1), round(y, 1), vis))
            if len(o["history"]) > 400:               # bound memory; keep first + recent
                o["history"] = o["history"][:1] + o["history"][-399:]
        self.last_step = int(step)

    # -- notes
    def add_note(self, step, state, facts, progress=""):
        self.notes.append({"step": int(step), "state": state.strip(), "facts": facts,
                           "progress": progress})
        if progress:
            self.progress = progress

    def active(self, role):
        """Most recent object entity carrying this role (pick/place targets change per sub-step)."""
        for o in reversed(list(self.objects.values())):
            if o["role"] == role:
                return o
        return None

    MAX_FACTS = 30

    def all_facts(self):
        """Task facts accumulated over the whole episode (deduplicated, oldest first, capped)."""
        facts = []
        for n in self.notes:
            for f in n["facts"]:
                if f not in facts and not f.startswith("step ") and "gripper release" not in f \
                        and not f.startswith("switched to") and "re-acquired" not in f:
                    facts.append(f)
        return facts[-self.MAX_FACTS:]

    def facts_text(self):
        facts = self.all_facts()
        return ("Known facts: " + "; ".join(facts) + ". ") if facts else ""

    # -- prompt context
    def context(self):
        lines = [f'Task: "{self.instruction}".']
        if self.plan:
            lines.append("Plan: " + " ".join(f"({i + 1}) {p}" for i, p in enumerate(self.plan)))
        if self.progress:
            lines.append(f"Last progress judgement: {self.progress}")
        lines.append("Known objects (id: name [role] at pixel x,y; visibility):")
        for o in self.objects.values():
            lines.append(f'  {o["id"]}: {o["name"]} [{o["role"]}] at '
                         f'({o["xy"][0]:.0f},{o["xy"][1]:.0f}); '
                         f'{"visible" if o["visible"] else "LOST by tracker"}')
        facts = self.all_facts()
        if facts:
            lines.append("Facts recorded so far: " + "; ".join(facts))
        if os.environ.get("MOLMO2_CTX_STATES", "0") == "1" and self.notes:
            lines.append("Previous observations (oldest first):")
            for n in self.notes[-MAX_NOTES_CTX:]:
                pr = f' | progress: {n["progress"]}' if n.get("progress") else ""
                lines.append(f'  step {n["step"]}: {n["state"]}{pr}')
        return "\n".join(lines)

    def dump(self):
        return {"session_id": self.session_id, "instruction": self.instruction,
                "width": self.W, "height": self.H, "last_step": self.last_step,
                "plan": self.plan, "progress": self.progress,
                "objects": self.objects, "notes": self.notes, "created": self.created}

    def persist(self):
        if not MEM_DIR:
            return
        os.makedirs(MEM_DIR, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(self.session_id))
        tmp = os.path.join(MEM_DIR, safe + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.dump(), f, indent=1)
        os.replace(tmp, os.path.join(MEM_DIR, safe + ".json"))


PLAN_PROMPT = ('This is a robot manipulation scene. The robot must complete the task: "{instr}". '
               'Break the task into an ordered list of sub-steps. Each step must say which object '
               'the gripper grasps and where it puts it (e.g. "grasp the left cup, put it over the '
               'red block"). Look at the image to see which objects are movable. Keep exactly the '
               'order the task states (e.g. "in the order: red, green, blue" means red first). '
               'One step per line, formatted "1. ...", "2. ...". Only list the steps.')

STEP_TARGETS_PROMPT = ('This is a robot manipulation scene. Task: "{instr}". {facts}Current sub-step: '
                       '"{step}". Which VISIBLE object should the robot grasp, and where should it '
                       'put it? Use the known facts — "completed: ..." facts say what was already '
                       'done, so they tell you where things are now (e.g. if "grasp the left cup, '
                       'put it over the green block" was completed, the green block is under the '
                       'left cup, so for "remove the cup from the green block" grasp "the left '
                       'cup"). Answer exactly: "grasp: <short noun phrase>; place: <short noun '
                       'phrase or none>".')

STEP_NAME_PROMPT = ('This is a robot manipulation scene. Task: "{instr}". {facts}Current sub-step: '
                    '"{step}". Which VISIBLE object should the robot {verb} for this sub-step? '
                    'Use the known facts — "completed: ..." facts say what was already done, so '
                    'they tell you where things are now (e.g. if "grasp the left cup, put it over the '
                    'green block" was completed, the green block is under the left cup, and the object '
                    'to grasp for "remove the cup from the green block" is "the left cup"). Answer '
                    'with a short noun phrase only, or "none" if there is no such object.')
POINT_PROMPT = 'Point to {phrase}.'

STEP_PICK_PROMPT = ('This is a robot manipulation scene. Task: "{instr}". {facts}Current sub-step: '
                    '"{step}". Point to the single object the robot should grasp/manipulate for this '
                    'sub-step (use the known facts if the object is hidden or covered).')
STEP_PLACE_PROMPT = ('This is a robot manipulation scene. Task: "{instr}". {facts}Current sub-step: '
                     '"{step}". Point to the location where the object should be placed / moved to for '
                     'this sub-step (use the known facts if the target is hidden or covered). If this '
                     'sub-step has no place target, answer "none".')

STEP_DONE_PROMPT = (
    'Task: "{instr}". The plan is: {plan}. The robot has been executing step {k}: "{step}". '
    'The images are consecutive frames, the last one is the current moment. Has step {k} been '
    'fully completed — i.e. the object of that step is now at its target location and the '
    'gripper has released it? Answer strictly "YES: <what you see that proves it>" or '
    '"NO: <what is still missing>".')

OBSERVE_BRIEF_PROMPT = (
    "You are the memory of a robot performing a manipulation task. {ctx}\n"
    "Look at the current image(s) (consecutive frames, last = now). Answer in exactly two "
    "short lines:\nSTATE: <one short sentence: what the gripper holds and where the "
    "task-relevant objects are>\nFACTS: <new facts worth remembering, semicolon-separated, "
    "or 'none'>")

OBSERVE_PROMPT = (
    "You are the memory of a robot performing a manipulation task. Below is what you "
    "already know; then look at the CURRENT image(s) and update the record. Judge only from "
    "what is visible now — do not assume progress that you cannot see. When several images "
    "are given they are consecutive frames ending at the current moment.\n\n{ctx}\n\n"
    "Answer in exactly this format (plain text, three lines):\n"
    "STATE: <one or two sentences: what the gripper is doing/holding, where the task-relevant "
    "objects are relative to each other and to the gripper>\n"
    "PROGRESS: <which plan steps are already completed and which step is being executed now, "
    "e.g. 'steps 1-2 done, executing step 3: ...' — cite the visible evidence in brackets, e.g. "
    "'(left cup already covers a block; gripper empty)'; say 'nothing done yet, executing step 1' "
    "at the start and 'task complete' when everything is done>\n"
    "FACTS: <semicolon-separated short facts worth remembering for this task, e.g. "
    "'red block is under the middle cup', 'left cup was lifted at step N'; write 'none' if "
    "nothing new>")

REACQ_PROMPT = ('This is a robot manipulation scene. The task is: "{instr}". '
                'The object "{name}" ({role} target) was previously at pixel '
                '({x:.0f},{y:.0f}) but the tracker lost it. Point to the {name} in the '
                'current image. If it is not visible, answer "not visible".')


def _parse_plan(txt):
    steps = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip() for l in txt.splitlines()
             if re.match(r"^\s*\d+[.)]", l)]
    return steps or [l.strip() for l in txt.splitlines() if l.strip()][:8]


def _current_step(mem):
    """Index (0-based) of the sub-step being executed according to the last PROGRESS line."""
    if not mem.plan or not mem.progress:
        return 0
    if "complete" in mem.progress.lower():
        return len(mem.plan) - 1
    m = re.search(r"execut\w*\s+step\s+(\d+)", mem.progress, re.I) or \
        re.search(r"step\s+(\d+)\s*(?:is|being|now|in progress|current)", mem.progress, re.I)
    if m:
        return int(np.clip(int(m.group(1)) - 1, 0, len(mem.plan) - 1))
    m = re.findall(r"steps?\s+(\d+)(?:\s*[-–]\s*(\d+))?\s+(?:are\s+)?done", mem.progress, re.I)
    if m:
        last = max(int(b or a) for a, b in m)
        return int(np.clip(last, 0, len(mem.plan) - 1))
    return 0


def _parse_observe(txt):
    state, facts, progress = "", [], ""
    m = re.search(r"STATE:\s*(.+?)(?:\n|$)", txt, re.S)
    if m:
        state = m.group(1).strip()
    m = re.search(r"PROGRESS:\s*(.+?)(?:\n|$)", txt, re.S)
    if m:
        progress = m.group(1).strip()
    m = re.search(r"FACTS:\s*(.+)", txt, re.S)
    if m:
        f = m.group(1).strip()
        if f.lower() not in ("none", "none.", ""):
            facts = [s.strip() for s in re.split(r";|\n", f) if s.strip()]
    if not state:
        state = txt.strip()[:400]
    return state, facts, progress


def session_start(session_id, instruction, img, want_place=True, want_other=False,
                  preset_objects=None):
    """preset_objects=[{"name","role","xy"}]: caller already localized the objects (e.g.
    per-slot /point queries) -> just register them (no pointing), still write note 0."""
    W, H = img.size
    mem = Memory(session_id, instruction, W, H)
    raws = {}
    if preset_objects:
        for o in preset_objects:
            mem.add_object(str(o.get("name", "object")), str(o.get("role", "other")),
                           (float(o["xy"][0]), float(o["xy"][1])))
    else:
        raw = _ask(img, PLAN_PROMPT.format(instr=instruction), max_new_tokens=256); raws["plan"] = raw
        mem.plan = _parse_plan(raw)
        step0 = mem.plan[0] if mem.plan else instruction
        raw = _ask(img, STEP_PICK_PROMPT.format(instr=instruction, step=step0, facts="")); raws["pick"] = raw
        for x, y in parse_points(raw, W, H)[:1]:
            mem.add_object(f"pick target (step 1: {step0})", "pick", (x, y))
        if want_place:
            raw = _ask(img, STEP_PLACE_PROMPT.format(instr=instruction, step=step0, facts="")); raws["place"] = raw
            for x, y in parse_points(raw, W, H)[:1]:
                mem.add_object(f"place target (step 1: {step0})", "place", (x, y))
        if want_other:
            raw = _ask(img, OTHER_PROMPT.format(instr=instruction)); raws["other"] = raw
            for i, (x, y) in enumerate(parse_points(raw, W, H)[:6]):
                mem.add_object(f"other object {i}", "other", (x, y))
    # frame-0 state note (brief unless MOLMO2_VERBOSE=1)
    if os.environ.get("MOLMO2_VERBOSE", "0") == "1":
        raw = _ask(img, OBSERVE_PROMPT.format(ctx=mem.context()), max_new_tokens=200)
    else:
        raw = _ask(img, OBSERVE_BRIEF_PROMPT.format(ctx=mem.context()), max_new_tokens=80)
    raws["observe0"] = raw
    st, facts, progress = _parse_observe(raw)
    mem.add_note(0, st, facts, progress)
    mem.step_idx = 0
    with _sess_lock:
        _sessions[session_id] = mem
    mem.persist()
    return {"session_id": session_id, "width": W, "height": H, "plan": mem.plan,
            "objects": list(mem.objects.values()), "state": st, "facts": facts,
            "progress": progress, "raw": raws}


def session_observe(session_id, step, img, tracked, history=None, events=None):
    """history: optional list of earlier PIL frames (oldest first) shown together with the
    current frame so the model can see the motion. events: {"releases": n} gripper release
    events since the last observe (drives sub-step progress, see MOLMO2_STEP_CHECK)."""
    with _sess_lock:
        mem = _sessions.get(session_id)
    if mem is None:
        raise KeyError(f"unknown session {session_id}")
    mem.update_tracked(step, tracked)
    W, H = img.size
    clip = (list(history) + [img]) if history else img
    mode = os.environ.get("MOLMO2_STEP_CHECK", "gripper")
    if mode == "vlm" or os.environ.get("MOLMO2_VERBOSE", "0") == "1":
        raw = _ask(clip, OBSERVE_PROMPT.format(ctx=mem.context()), max_new_tokens=200)
    else:                                   # brief: one-line state, no progress narrative
        raw = _ask(clip, OBSERVE_BRIEF_PROMPT.format(ctx=mem.context()), max_new_tokens=80)
    st, facts, progress = _parse_observe(raw)
    mem.add_note(step, st, facts, progress if mode == "vlm" else "")   # narrative progress is
    # only authoritative in vlm mode; in gripper mode the release events decide (below)
    # step control. Default ("gripper"): the caller reports gripper release events since the
    # last observe (each pick-and-place sub-step ends with a release) — reliable on a real
    # robot. "vlm": ask the model whether the current sub-step is complete (binary, with
    # evidence). "both": gripper events advance, the model may additionally advance.
    new_targets = []
    k = getattr(mem, "step_idx", 0)
    n_rel = int(events.get("releases", 0)) if events else 0
    if mem.plan and mode in ("gripper", "both") and n_rel > 0:
        k0 = k
        k = min(k + n_rel, len(mem.plan) - 1)
        if k == len(mem.plan) - 1 and getattr(mem, "step_idx", 0) + n_rel > len(mem.plan) - 1:
            mem.progress = "task complete"
        # deterministic memory of what has been done: the completed steps' own wording tells
        # later prompts where things are now (e.g. "the left cup is over the green block")
        done_upto = min(k0 + n_rel, len(mem.plan))
        for j in range(k0, done_upto):
            mem.notes[-1]["facts"].append(f"completed: {mem.plan[j]}")
        mem.notes[-1]["facts"].append(f"{n_rel} gripper release(s) -> advanced to step {k + 1}")
    if mem.plan and mode in ("vlm", "both"):
        for _ in range(2):
            if k >= len(mem.plan) - 1 and mem.progress.lower().startswith("task complete"):
                break
            r = _ask(clip, STEP_DONE_PROMPT.format(instr=mem.instruction, k=k + 1, step=mem.plan[k],
                                                   plan=" ".join(f"({i + 1}) {p}" for i, p in enumerate(mem.plan))),
                     max_new_tokens=96)
            done = r.strip().upper().startswith("YES")
            mem.notes[-1]["facts"].append(f"step {k + 1} check: {r.strip()[:120]}")
            if not done:
                break
            if k >= len(mem.plan) - 1:
                mem.progress = "task complete"; mem.notes[-1]["progress"] = mem.progress
                break
            k += 1
        mem.progress = ("task complete" if mem.progress == "task complete" else
                        f"steps 1-{k} done, executing step {k + 1}: {mem.plan[k]}" if k > 0 else
                        f"nothing done yet, executing step 1: {mem.plan[0]}")
        mem.notes[-1]["progress"] = mem.progress
        progress = mem.progress
    if mem.plan and mode in ("gripper", "both") and mode != "vlm":
        if mem.progress != "task complete":
            mem.progress = (f"steps 1-{k} done, executing step {k + 1}: {mem.plan[k]}" if k > 0 else
                            f"nothing done yet, executing step 1: {mem.plan[0]}")
        mem.notes[-1]["progress"] = mem.progress
        progress = mem.progress
    if mem.plan and k != getattr(mem, "step_idx", 0):
        mem.step_idx = k
        stxt = mem.plan[k]
        # two-stage: ONE question names the grasp / place objects (reasoning over the memory
        # facts), then one pointing query per role
        ans = _ask(img, STEP_TARGETS_PROMPT.format(instr=mem.instruction, step=stxt,
                                                  facts=mem.facts_text()), max_new_tokens=48)
        names = {}
        for role_key, pat in (("pick", r"grasp\s*:\s*([^;\n]+)"), ("place", r"place\s*:\s*([^;\n]+)")):
            m = re.search(pat, ans, re.I)
            names[role_key] = m.group(1).strip().strip('."') [:60] if m else ""
        for role, prompt in (("pick", STEP_PICK_PROMPT), ("place", STEP_PLACE_PROMPT)):
            name = names.get(role, "")
            p = []
            if name and name.lower() != "none":
                r = _ask(img, POINT_PROMPT.format(phrase=name))
                p = parse_points(r, W, H)
            if not p:                                     # fallback: direct pointing prompt
                r = _ask(img, prompt.format(instr=mem.instruction, step=stxt, facts=mem.facts_text()))
                p = parse_points(r, W, H)
            print(f"[planner]   retarget step {k + 1} {role}: name={name!r} -> {p[:1]} raw={r.strip()[:80]!r}",
                  flush=True)
            if p:
                for o in mem.objects.values():
                    if o["role"] == role:
                        o["role"] = "done"
                oid = mem.add_object(f"{role} target (step {k + 1}: {stxt})", role, p[0], step)
                new_targets.append({"id": oid, "role": role, "xy": mem.objects[oid]["xy"]})
        mem.notes[-1]["facts"].append(f"switched to step {k + 1}: {stxt}")
    reacq = []
    for o in mem.objects.values():
        if o["visible"] or o["role"] == "other":
            continue
        r = _ask(img, REACQ_PROMPT.format(instr=mem.instruction, name=o["name"],
                                          role=o["role"], x=o["xy"][0], y=o["xy"][1]))
        p = parse_points(r, W, H)
        if p:
            o["xy"], o["visible"] = [round(p[0][0], 1), round(p[0][1], 1)], True
            o["history"].append((int(step), o["xy"][0], o["xy"][1], True))
            reacq.append({"id": o["id"], "xy": o["xy"]})
            mem.notes[-1]["facts"].append(f'{o["name"]} re-acquired by planner at step {step}')
    mem.persist()
    return {"session_id": session_id, "step": int(step), "state": st, "facts": facts,
            "progress": progress, "step_index": getattr(mem, "step_idx", 0), "plan": mem.plan,
            "reacquired": reacq, "new_targets": new_targets,
            "objects": list(mem.objects.values()), "raw": raw}


# ----------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "model": MODEL_DIR, "loaded": _model is not None,
                             "sessions": len(_sessions)})
        elif u.path.rstrip("/") == "/memory":
            sid = (parse_qs(u.query).get("session_id") or [""])[0]
            with _sess_lock:
                mem = _sessions.get(sid)
            if mem is None:
                return self._send(404, {"ok": False, "err": f"unknown session {sid!r}"})
            self._send(200, mem.dump())
        else:
            self._send(404, {"ok": False})

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            path = self.path.rstrip("/")
            t0 = time.time()
            if path == "/plan":
                out = plan(_decode_image(req), str(req.get("instruction", "")),
                           want_place=bool(req.get("want_place", True)))
                print(f"[planner] /plan instr={req.get('instruction', '')!r} -> "
                      f"pick={out['pick']} place={out['place']} ({time.time() - t0:.1f}s)",
                      flush=True)
            elif path == "/point":
                out = point_queries(_decode_image(req), list(req.get("queries", [])),
                                    int(req.get("max_new_tokens", 192)))
                print(f"[planner] /point x{len(out['results'])}: "
                      + "; ".join(f"{r['name']}={r['points'][:1]}" for r in out["results"])
                      + f" ({time.time() - t0:.1f}s)", flush=True)
            elif path == "/session":
                out = session_start(str(req["session_id"]), str(req.get("instruction", "")),
                                    _decode_image(req), want_place=bool(req.get("want_place", True)),
                                    want_other=bool(req.get("want_other", False)),
                                    preset_objects=req.get("objects"))
                print(f"[planner] /session {req['session_id']} instr={req.get('instruction', '')!r}"
                      f" plan={out['plan']} -> objects={[(o['name'], o['role'], o['xy']) for o in out['objects']]}"
                      f" | STATE: {out['state']} | PROGRESS: {out['progress']} ({time.time() - t0:.1f}s)", flush=True)
            elif path == "/observe":
                hist = [_decode_image({"image_b64": b}) for b in req.get("history_b64", [])]
                out = session_observe(str(req["session_id"]), int(req.get("step", 0)),
                                      _decode_image(req), list(req.get("tracked", [])), history=hist,
                                      events=req.get("events"))
                print(f"[planner] /observe step={out['step']} {out['progress']} | {out['state'][:80]}"
                      + (f" | new={[(n['role'], n['xy']) for n in out['new_targets']]}" if out['new_targets'] else "")
                      + f" ({time.time() - t0:.1f}s)", flush=True)
            else:
                return self._send(404, {"ok": False, "err": "unknown endpoint"})
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send(500, {"ok": False, "err": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):  # quiet default access log; we print our own lines
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", default="", help="image path: run /session logic offline")
    ap.add_argument("--instruction", default="Pick up the object on the table")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    if args.selftest:
        from PIL import Image
        _load()
        img = Image.open(args.selftest).convert("RGB")
        out = session_start("selftest", args.instruction, img)
        print(json.dumps(out, indent=2))
        # second look at the same image with fake tracker feedback exercises /observe
        tracked = [{"id": o["id"], "xy": o["xy"], "visible": True} for o in out["objects"]]
        print(json.dumps(session_observe("selftest", 30, img, tracked), indent=2))
        return
    _load()                                   # load BEFORE serving: /health loaded=true
    # warm-up: the first generation after loading is several times slower (kernel/graph
    # setup); do it on a dummy image so the first real episode does not pay for it
    from PIL import Image
    t0 = time.time()
    _ask(Image.new("RGB", (640, 480), (90, 90, 120)), "Point to the object.", max_new_tokens=16)
    print(f"[planner] warm-up done in {time.time() - t0:.0f}s", flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"MOLMO2_PLANNER_SERVING port={args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
