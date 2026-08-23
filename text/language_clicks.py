"""instruction -> T5ObjectSegmenter phrases -> grounding language interface -> pixel clicks.

The text module's output IS the grounding query: each extracted object phrase is sent to
the pointing service (`gam.planner.PlannerClient.point`, one "Point to the <phrase>."
query per unique phrase), and the returned pixels become the slots' clicks (frame 0, held
constant — same contract as recorded frame-0 clicks).

config:
    clicks:
      source: language
      grounding_url: http://127.0.0.1:18092     # gam/planner/server.py
      segmenter_ckpt: /path/t5_segmenter.pt     # text/train_segmenter.py output

Phrase -> slot assignment: stemmed word overlap with the slot name first (battery_slot <->
"battery slot", battery1 <-> "batteries"), remaining phrases in reading order; a slot with
nothing left falls back to its own cleaned name, so under-segmented instructions degrade,
not crash. When one phrase serves several slots ("batteries" -> battery1, battery2) the
service's multiple points are handed out left-to-right.
"""
import re

import numpy as np

_seg = _cli = None


def get_segmenter(cfg):
    """Load (once) the T5 segmenter from config: top-level `text:` section, falling back to
    the clicks section. One shared instance serves both branches — object spans for
    grounding clicks and the whole-sentence embedding for the CondFuse language token."""
    global _seg
    if _seg is None:
        from gam.text import T5ObjectSegmenter
        t = dict(cfg.get("text") or {})
        c = cfg["data"].get("clicks", {})
        ckpt = t.get("segmenter_ckpt") or c.get("segmenter_ckpt")
        assert ckpt, "text.segmenter_ckpt (or clicks.segmenter_ckpt) required for the language pathway"
        _seg = T5ObjectSegmenter.from_checkpoint(
            ckpt, cache_dir=t.get("hf_cache_dir") or c.get("hf_cache_dir"))
    return _seg


def sentence_embedding(cfg, instruction):
    """Whole-sentence T5 embedding (768,) — the CondFuse language token's input."""
    return get_segmenter(cfg).sentence_embedding(instruction or "")


def _stem(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _clean(s):
    return re.sub(r"[_\d]+", " ", s.lower()).split()


def _words(s):
    return set(_stem(w) for w in _clean(s) if w)


def assign_phrases(slots, phrases):
    out = {}
    pairs = sorted(((len(_words(sl) & _words(ph)), -len(_words(ph)), i, j)
                    for i, sl in enumerate(slots) for j, ph in enumerate(phrases)),
                   reverse=True)
    used = set()
    owner = {}                                     # phrase j -> word set of first taker
    for ov, _, i, j in pairs:                      # 1) name overlap, best first
        if ov <= 0 or i in out:
            continue
        if j in owner and owner[j] != _words(slots[i]):
            continue                               # reuse only for twin slots (battery1/battery2)
        out[i] = phrases[j]; used.add(j); owner[j] = _words(slots[i])
    left = [j for j in range(len(phrases)) if j not in used]
    for i in range(len(slots)):                    # 2) reading order for the rest
        if i not in out and left:
            out[i] = phrases[left.pop(0)]
    return [out.get(i) or " ".join(_clean(slots[i])) for i in range(len(slots))]


def clicks_from_language(cfg, rgb0, instruction):
    """-> ((S,2) float32 px  (NaN where the service found nothing), [query per slot])"""
    global _cli
    c = cfg["data"]["clicks"]
    seg = get_segmenter(cfg)
    if _cli is None:
        from gam.planner.client import PlannerClient
        _cli = PlannerClient(c.get("grounding_url", "http://127.0.0.1:18092"))
    slots = cfg["data"]["slots"]
    phrases = [g["text"] for g in seg.extract(instruction or "")]
    queries = assign_phrases(slots, phrases)
    uniq = sorted(set(queries))
    r = _cli.point(rgb0, [{"name": q, "text": f"Point to the {q}."} for q in uniq])
    pts = {res["name"]: list(res.get("points", [])) for res in r.get("results", [])}
    for q in pts:                                  # deterministic multi-instance order
        pts[q].sort(key=lambda p: p[0])
    px = np.full((len(slots), 2), np.nan, np.float32)
    for i, q in enumerate(queries):                # same phrase -> next unclaimed instance
        if pts.get(q):
            px[i] = np.asarray(pts[q].pop(0)[:2], np.float32)
    return px, queries
