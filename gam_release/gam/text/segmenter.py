"""Object-word segmenter on a frozen T5 text encoder.

    instruction --T5 encoder--> per-token features --small BIO head--> object spans
    "pick up bowl"  ->  [{"text": "bowl", "span": (8, 12), "emb": (d_model,) float32}]

The T5 encoder is FROZEN — only the 3-way (O/B/I) tagging head trains
(`yam/text/train_segmenter.py`, weak labels from instruction templates x an object
vocabulary). `emb` is the mean of the span's T5 token features: use the surface text as a
grounding query (WildDet3D text prompt, a pointing service, ...) and/or the embedding as a
language token for the policy.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class T5ObjectSegmenter(nn.Module):
    def __init__(self, t5_name: str = "google/flan-t5-base", cache_dir: Optional[str] = None,
                 device: str = "cuda", head_hidden: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel
        self.t5_name = t5_name
        self.tok = AutoTokenizer.from_pretrained(t5_name, cache_dir=cache_dir)
        self.enc = T5EncoderModel.from_pretrained(t5_name, cache_dir=cache_dir).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        d = self.enc.config.d_model
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(), nn.Linear(head_hidden, 3))
        self.dev = device
        self.to(device)

    # (B,L,d) frozen features + attention mask + char offsets per token
    def encode(self, texts: List[str]):
        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
                     return_offsets_mapping=True)
        off = b.pop("offset_mapping")
        b = {k: v.to(self.dev) for k, v in b.items()}
        with torch.no_grad():
            h = self.enc(**b).last_hidden_state
        return h, b["attention_mask"], off

    def forward(self, texts: List[str]):
        h, m, off = self.encode(texts)
        return self.head(h), m, off, h

    @torch.no_grad()
    def sentence_embedding(self, instruction: str):
        """Whole-sentence embedding: mean of the T5 encoder's valid token features -> np (d,)."""
        h, m, _ = self.encode([instruction or ""])
        w = m[0].bool()
        return h[0][w].mean(0).float().cpu().numpy()

    @torch.no_grad()
    def extract_full(self, instruction: str):
        """One T5 forward, two branches: (object spans, whole-sentence embedding)."""
        logits, m, off, h = self.forward([instruction])
        w = m[0].bool()
        return self._spans(instruction, logits, m, off, h), h[0][w].mean(0).float().cpu().numpy()

    @torch.no_grad()
    def extract(self, instruction: str):
        """-> [{"text", "span": (char0, char1), "emb": np (d,)}, ...] in reading order."""
        logits, m, off, h = self.forward([instruction])
        return self._spans(instruction, logits, m, off, h)

    def _spans(self, instruction, logits, m, off, h):
        L = int(m[0].sum())
        tags = logits[0, :L].argmax(-1).tolist()
        off0 = off[0].tolist()
        spans, cur = [], None
        for i in range(L):
            t = 0 if off0[i][1] == 0 else tags[i]          # specials (</s>) are O
            if t == 1:
                if cur is not None:
                    spans.append(cur)
                cur = [i, i]
            elif t == 2 and cur is not None:
                cur[1] = i
            else:
                if cur is not None:
                    spans.append(cur)
                cur = None
        if cur is not None:
            spans.append(cur)
        out = []
        for i0, i1 in spans:
            c0, c1 = int(off0[i0][0]), int(off0[i1][1])
            while c0 < c1 and instruction[c0].isspace():   # sentencepiece offsets include the leading space
                c0 += 1
            out.append({"text": instruction[c0:c1], "span": (c0, c1),
                        "emb": h[0, i0:i1 + 1].mean(0).float().cpu().numpy()})
        return out

    def save_head(self, path: str):
        torch.save({"t5_name": self.t5_name, "head": self.head.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path: str, cache_dir: Optional[str] = None, device: str = "cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = cls(ck["t5_name"], cache_dir=cache_dir, device=device)
        m.head.load_state_dict(ck["head"])
        return m
