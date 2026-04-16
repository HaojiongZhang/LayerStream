from __future__ import annotations

import torch


class StreamManager:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this prototype.")
        self.compute_stream = torch.cuda.default_stream()
        self.copy_stream = torch.cuda.Stream()
        self.kv_stream = torch.cuda.Stream()