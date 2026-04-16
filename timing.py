from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class WallTimer:
    t0: float = 0.0
    t1: float = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0


class CudaTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)

    def start(self, stream: torch.cuda.Stream | None = None) -> None:
        if not self.enabled:
            return
        if stream is None:
            self.start_evt.record()
        else:
            self.start_evt.record(stream)

    def stop(self, stream: torch.cuda.Stream | None = None) -> None:
        if not self.enabled:
            return
        if stream is None:
            self.end_evt.record()
        else:
            self.end_evt.record(stream)

    def elapsed_ms(self) -> float:
        if not self.enabled:
            return 0.0
        torch.cuda.synchronize()
        return float(self.start_evt.elapsed_time(self.end_evt))