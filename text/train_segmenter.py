"""Train the O/B/I object-span head of gam.text.T5ObjectSegmenter with weak labels.

    python text/train_segmenter.py [--out t5_segmenter.pt] [--extra-vocab words.txt]
                                   [--steps 1500] [--t5 google/flan-t5-base]

Data is synthesized: instruction templates x an object vocabulary; the object noun phrase's
character span is known by construction and projected onto T5 subword tokens via offsets.
The vocabulary is split train/test so the reported span accuracy is on UNSEEN object words —
the head must generalize from T5's representation, not memorize the nouns.
Add your domain's words (one per line, multi-word ok) with --extra-vocab.
"""
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

OBJECTS = """bowl cup mug plate bottle battery block button cover drawer door spoon fork knife
apple banana orange box lid jar can hammer screwdriver brush sponge towel shoe book pen pencil
phone remote cube ball basket tray pot pan kettle teapot scissors tape stapler marker eraser
charger cable mouse keyboard laptop tablet glass pitcher carrot tomato potato onion bread egg
key lock clip clamp wrench pliers drill nail screw bolt flashlight candle lighter vase flower
coin card dice lego doll duck train truck airplane boat helmet glove sock hat scarf belt wallet
brick sphere cylinder cone ring peg gear lever handle knob switch plug socket funnel whisk
ladle spatula grater peeler opener corkscrew straw napkin coaster placemat saucer chopstick
battery slot coffee cup tea cup wine glass milk carton soda can trash bin plant pot toy car
power bank remote control cutting board measuring cup rolling pin oven mitt paper towel""".split("\n")
OBJECTS = [w for line in OBJECTS for w in ([line] if " " in line and len(line.split()) <= 3 and "\n" not in line else line.split())]
OBJECTS = sorted(set(w.strip() for chunk in OBJECTS for w in ([chunk] if " " not in chunk else [chunk])
                     if w.strip()))
MULTI = ["battery slot", "coffee cup", "tea cup", "wine glass", "milk carton", "soda can",
         "trash bin", "plant pot", "toy car", "power bank", "remote control", "cutting board",
         "measuring cup", "rolling pin", "oven mitt", "paper towel"]
OBJECTS = sorted(set([o for o in OBJECTS if o not in " ".join(MULTI).split()]) | set(MULTI))
COLORS = ["red", "green", "blue", "yellow", "pink", "purple", "orange", "white", "black", "gray"]

T1 = ["pick up the {o}", "pick up {o}", "grab the {o}", "lift the {o}", "press the {o}",
      "push the {o}", "open the {o}", "close the {o}", "shake the {o}", "rotate the {o}",
      "hand over the {o}", "point to the {o}", "hold the {o} steady", "slide the {o} forward",
      "flip the {o} over", "tap the {o} twice", "please pass me the {o}",
      "carefully pick up the {o} without tipping it"]
T2 = ["put the {o} into the {p}", "place the {o} on the {p}", "stack the {o} on the {p}",
      "move the {o} next to the {p}", "cover the {o} with the {p}", "take the {o} out of the {p}",
      "insert the {o} into the {p}", "pour water from the {o} into the {p}",
      "put the {o} and the {q} into the {p}", "swap the {o} and the {q}",
      "first pick up the {o} then drop it in the {p}"]
TD = ["there are two {o}s and a {p} on the table", "there is a {o} next to the {p}",
      "you can see a {o} and a {p}", "a {o} sits inside the {p}",
      "two {o}s are lying near the {p}", "the {o} is on top of the {p}",
      "on the table are a {o}, a {p} and a {q}",
      "combining the two {o}s in different orientations moves the {p}"]
T0 = ["move to the left", "go back home", "stop now", "wait a moment", "raise both arms",
      "move a bit closer", "turn around", "do nothing"]


def phrase(o, rng):
    return f"{rng.choice(COLORS)} {o}" if rng.random() < 0.35 else o


def sample(rng, vocab):
    kind = rng.random()
    if kind < 0.08:
        return rng.choice(T0), []
    objs = [phrase(o, rng) for o in rng.sample(vocab, 3)]
    if kind < 0.45:
        t = rng.choice(T1); s = t.format(o=objs[0]); keep = [objs[0]]
    elif kind < 0.70:
        t = rng.choice(TD); s = t.format(o=objs[0], p=objs[1], q=objs[2])
        keep = []
        for x in objs:
            for form in (x + "s", x):              # plural surface form first (longer match)
                if form in s:
                    keep.append(form); break
    else:
        t = rng.choice(T2); s = t.format(o=objs[0], p=objs[1], q=objs[2])
        keep = [x for x in objs if ("{q}" in t or x != objs[2]) and x in s]
    spans = []
    for ph in keep:
        i = s.find(ph)
        if i >= 0:
            spans.append((i, i + len(ph)))
    return s, sorted(spans)


def tags_from_spans(off, spans, L):
    y = np.zeros(L, np.int64)
    for (c0, c1) in spans:
        first = True
        for i in range(L):
            o0, o1 = off[i]
            if o1 == 0:
                continue
            if o0 < c1 and o1 > c0:
                y[i] = 1 if first else 2
                first = False
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="t5_segmenter.pt")
    ap.add_argument("--extra-vocab", default="")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--t5", default="google/flan-t5-base")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    from gam.text import T5ObjectSegmenter
    rng = random.Random(0)
    vocab = list(OBJECTS)
    if args.extra_vocab:
        vocab += [l.strip() for l in open(args.extra_vocab) if l.strip()]
    vocab = sorted(set(vocab))
    rng.shuffle(vocab)
    n_test = max(8, len(vocab) // 7)
    test_vocab, train_vocab = vocab[:n_test], vocab[n_test:]
    if "bowl" in train_vocab:                       # the canonical demo word stays unseen
        train_vocab.remove("bowl"); test_vocab.append("bowl")
    print(f"[seg] vocab train={len(train_vocab)} test(unseen)={len(test_vocab)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ObjectSegmenter(args.t5, cache_dir=args.cache_dir, device=dev)
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    B = 48
    for step in range(1, args.steps + 1):
        texts, spans = zip(*[sample(rng, train_vocab) for _ in range(B)])
        logits, m, off, _ = model(list(texts))
        L = logits.shape[1]
        y = torch.stack([torch.from_numpy(tags_from_spans(off[i].tolist(), spans[i], L))
                         for i in range(B)]).to(dev)
        w = m.bool() & torch.tensor([[o[1] > 0 for o in off[i].tolist()] for i in range(B)], device=dev)
        loss = F.cross_entropy(logits[w], y[w])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
    # ---- exact-span accuracy on UNSEEN vocab ----
    rng2 = random.Random(1); hit = tot = 0
    for _ in range(300):
        s, spans = sample(rng2, test_vocab)
        got = [(g["span"][0], g["span"][1]) for g in model.extract(s)]
        tot += 1; hit += int(got == spans)
    print(f"[seg] exact-span accuracy on unseen objects: {hit}/{tot} = {hit / tot * 100:.1f}%")
    for demo in ("pick up bowl", "pick up the bowl", "put the red block into the battery slot",
                 "press the button then grab the coffee cup"):
        print(f"  '{demo}' ->", [g["text"] for g in model.extract(demo)])
    model.save_head(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
