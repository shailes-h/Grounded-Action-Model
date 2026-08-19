"""Training profiler for performance analysis.

Usage:
    Set environment variable PROFILE_WILDDET3D=1 to enable profiling.
    Timing results are printed every N iterations.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


class TrainingProfiler:
    """Profiler for measuring training component timings."""

    _instance: Optional["TrainingProfiler"] = None

    def __init__(self, print_interval: int = 10, enabled: bool = True):
        self.print_interval = print_interval
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0
        self.current_step_timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "TrainingProfiler":
        """Get singleton instance."""
        if cls._instance is None:
            enabled = os.environ.get("PROFILE_WILDDET3D", "0") == "1"
            print_interval = int(os.environ.get("PROFILE_INTERVAL", "10"))
            cls._instance = cls(print_interval=print_interval, enabled=enabled)
            if enabled:
                print(f"[TrainingProfiler] Enabled, printing every {print_interval} steps")
        return cls._instance

    def _is_main_process(self) -> bool:
        import multiprocessing
        current = multiprocessing.current_process()
        return current.name == "MainProcess"

    def _safe_cuda_sync(self) -> None:
        if self._is_main_process() and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if not self._is_main_process():
            return
        self._safe_cuda_sync()
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        if not self._is_main_process():
            return 0.0
        self._safe_cuda_sync()
        elapsed = time.perf_counter() - self._start_times.get(name, time.perf_counter())
        self.current_step_timings[name] = elapsed
        return elapsed

    def step(self) -> None:
        if not self.enabled:
            return
        for name, elapsed in self.current_step_timings.items():
            self.timings[name].append(elapsed)
        self.step_count += 1

    def _is_rank_zero(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0


def profiler() -> TrainingProfiler:
    """Get the global profiler instance."""
    return TrainingProfiler.get_instance()


def profile_start(name: str) -> None:
    """Start profiling a named section."""
    TrainingProfiler.get_instance().start(name)


def profile_stop(name: str) -> float:
    """Stop profiling a named section and return elapsed time."""
    return TrainingProfiler.get_instance().stop(name)


def profile_step() -> None:
    """Mark end of training step."""
    TrainingProfiler.get_instance().step()
