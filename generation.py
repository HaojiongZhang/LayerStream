from __future__ import annotations

from dataclasses import dataclass

import torch

from .forward_engine import ForwardEngine
from .kv_manager import KVCacheManager
from .logger import ExperimentLogger
from .timing import WallTimer


@dataclass
class GenerationResult:
    text: str
    generated_ids: list[int]


class Generator:
    def __init__(
        self,
        tokenizer,
        engine: ForwardEngine,
        kv_manager: KVCacheManager,
        logger: ExperimentLogger,
        eos_token_id: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self.kv_manager = kv_manager
        self.logger = logger
        self.eos_token_id = eos_token_id

    @staticmethod
    def _gpu_mem_mb() -> tuple[float, float]:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return allocated, reserved

    def generate(
        self,
        request_id: int,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> GenerationResult:
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")

        generated_ids: list[int] = []
        prompt_len = int(input_ids.shape[1])

        self.logger.log_request(
            request_id=request_id,
            event="start",
            prompt=prompt,
            prompt_len=prompt_len,
        )

        for token_idx in range(max_new_tokens):
            token_timer = WallTimer()
            token_timer.start()

            out = self.engine.forward_token(
                request_id=request_id,
                token_idx=token_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
            )

            next_token = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated_ids.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], device="cuda", dtype=input_ids.dtype)
            input_ids = next_token_tensor
            if attention_mask is not None:
                attention_mask = torch.ones((input_ids.shape[0], 1), device="cuda", dtype=attention_mask.dtype)

            token_timer.stop()
            allocated_mb, reserved_mb = self._gpu_mem_mb()

            self.logger.log_token(
                request_id=request_id,
                token_idx=token_idx,
                context_len=prompt_len + len(generated_ids),
                token_latency_ms=token_timer.elapsed_ms,
                copy_total_ms=out.total_copy_ms,
                compute_total_ms=out.total_compute_ms,
                wait_total_ms=out.total_wait_ms,
                gpu_allocated_mb=allocated_mb,
                gpu_reserved_mb=reserved_mb,
                kv_gpu_bytes=self.kv_manager.total_gpu_kv_bytes(),
                kv_cpu_bytes=self.kv_manager.total_cpu_kv_bytes(),
            )

            self.logger.log_memory(
                request_id=request_id,
                token_idx=token_idx,
                gpu_allocated_mb=allocated_mb,
                gpu_reserved_mb=reserved_mb,
                max_allocated_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
                max_reserved_mb=torch.cuda.max_memory_reserved() / (1024 ** 2),
            )

            if self.eos_token_id is not None and next_token == self.eos_token_id:
                break

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        self.logger.log_request(
            request_id=request_id,
            event="end",
            generated_tokens=len(generated_ids),
            final_text=text,
        )
        return GenerationResult(text=text, generated_ids=generated_ids)