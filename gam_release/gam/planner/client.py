"""HTTP client for the Molmo2 planner service (gam.planner.server)."""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import Dict, List, Optional

import cv2
import numpy as np


class PlannerClient:
    def __init__(self, url: str = "http://127.0.0.1:18092", timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never via proxy
        return json.loads(opener.open(req, timeout=self.timeout).read())

    @staticmethod
    def _jpeg_b64(rgb: np.ndarray, quality: int = 92) -> str:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        assert ok
        return base64.b64encode(enc.tobytes()).decode()

    def health(self) -> dict:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return json.loads(opener.open(self.url + "/health", timeout=10).read())

    def start_session(self, session_id: str, instruction: str, rgb: np.ndarray,
                      want_place: bool = True, want_other: bool = False,
                      objects: Optional[List[dict]] = None) -> dict:
        """Frame-0 plan. Returns objects [{"id","name","role","xy"}], state, facts."""
        payload = {"session_id": session_id, "instruction": instruction,
                   "image_b64": self._jpeg_b64(rgb), "want_place": want_place,
                   "want_other": want_other}
        if objects:
            payload["objects"] = objects
        return self._post("/session", payload)

    def observe(self, session_id: str, step: int, rgb: np.ndarray, tracked: List[Dict],
                history: Optional[List[np.ndarray]] = None, events: Optional[Dict] = None) -> dict:
        """Feed tracked object pixels back; planner judges progress, writes a state note and
        re-points lost / new targets. tracked: [{"id","xy":[u,v],"visible":bool}];
        history: optional earlier frames (oldest first) so the model sees the motion."""
        payload = {"session_id": session_id, "step": int(step),
                   "image_b64": self._jpeg_b64(rgb), "tracked": tracked}
        if history:
            payload["history_b64"] = [self._jpeg_b64(h, 80) for h in history]
        if events:
            payload["events"] = events            # e.g. {"releases": 1}
        return self._post("/observe", payload)

    def point(self, rgb: np.ndarray, queries: List[Dict[str, str]]) -> dict:
        return self._post("/point", {"image_b64": self._jpeg_b64(rgb), "queries": queries})

    def memory(self, session_id: str) -> dict:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return json.loads(opener.open(f"{self.url}/memory?session_id={session_id}", timeout=10).read())


class MockPlanner:
    """Offline stand-in for tests: fixed pixel points instead of Molmo2.
    spec: "mock:u,v[;u,v]" -> first point = pick, second = place."""

    def __init__(self, points: List[List[float]]):
        self.points = points

    @classmethod
    def from_spec(cls, spec: str) -> "MockPlanner":
        body = spec.split(":", 1)[1]
        pts = [[float(v) for v in p.split(",")] for p in body.split(";") if p.strip()]
        return cls(pts)

    def start_session(self, session_id, instruction, rgb, want_place=True, want_other=True, objects=None):
        roles = ["pick", "place"]
        objs = [{"id": f"o{i}", "name": f"{roles[i] if i < 2 else 'other'} target",
                 "role": roles[i] if i < 2 else "other", "xy": p}
                for i, p in enumerate(self.points) if (i == 0 or want_place or i > 1)]
        return {"session_id": session_id, "objects": objs, "state": "mock planner", "facts": []}

    def observe(self, session_id, step, rgb, tracked):
        return {"state": "mock planner", "facts": [], "reacquired": []}
