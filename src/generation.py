from __future__ import annotations

from dataclasses import dataclass

import torch

from .forward_engine import ForwardEngine
from .kv_manager import KVCacheManager
from .logger import ExperimentLogger
from .timing import WallTimer


@dataclass
class GenerationResult:
    """``texts`` / ``generated_ids`` always length ``batch_size``."""

    texts: list[str]
    generated_ids: list[list[int]]

    @property
    def text(self) -> str:
        """Single-row backward compatibility, or joined multi-row preview."""
        if len(self.texts) == 1:
            return self.texts[0]
        return "\n---\n".join(f"[{i}] {t}" for i, t in enumerate(self.texts))


class Generator:
    def __init__(
        self,
        tokenizer,
        engine: ForwardEngine,
        kv_manager: KVCacheManager,
        logger: ExperimentLogger,
        eos_token_id: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self.kv_manager = kv_manager
        self.logger = logger
        self.eos_token_id = eos_token_id
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size

    def logits_after_prompt(
        self,
        request_id: int,
        prompt: str,
        *,
        token_idx: int = 0,
    ) -> torch.Tensor:
        """
        Single prompt: run one forward and return logits at the last prompt
        position (shape ``[vocab]``).
        """
        self.kv_manager.reset_sequence()
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")

        out = self.engine.forward_token(
            request_id=request_id,
            token_idx=token_idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
        )
        torch.cuda.synchronize()
        return out.logits[0, -1, :].detach()

    def logits_after_prompts(
        self,
        request_id: int,
        prompts: list[str],
        *,
        token_idx: int = 0,
    ) -> torch.Tensor:
        """
        Batched prompts (possibly different lengths): right-padded tokenization,
        one forward, logits at the last **non-pad** position per row. Returns
        ``[B, vocab]``.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty")
        self.kv_manager.reset_sequence()
        tok = self.tokenizer
        if tok.pad_token_id is None and len(prompts) > 1:
            tok.pad_token_id = tok.eos_token_id
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device="cuda")
        else:
            attention_mask = attention_mask.to("cuda")

        out = self.engine.forward_token(
            request_id=request_id,
            token_idx=token_idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
        )
        torch.cuda.synchronize()
        logits_lm = out.logits
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        return logits_lm[b_idx, last_idx, :].detach()

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
        batch_size: int | None = None,
    ) -> GenerationResult:
        bs = self.batch_size if batch_size is None else batch_size
        if bs < 1:
            raise ValueError(f"batch_size must be >= 1, got {bs}")

        prompts = [prompt] * bs
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device="cuda")
        else:
            attention_mask = attention_mask.to("cuda")

        batch = int(input_ids.shape[0])
        per_row_ids: list[list[int]] = [[] for _ in range(batch)]
        prompt_len = int(input_ids.shape[1])

        self.logger.log_request(
            request_id=request_id,
            event="start",
            prompt=prompt if batch == 1 else f"{prompt!r} (x{batch})",
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

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            for b in range(batch):
                per_row_ids[b].append(int(next_token[b].item()))

            input_ids = next_token.unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = torch.ones(
                    (batch, 1), device="cuda", dtype=attention_mask.dtype
                )

            token_timer.stop()
            allocated_mb, reserved_mb = self._gpu_mem_mb()

            self.logger.log_token(
                request_id=request_id,
                token_idx=token_idx,
                context_len=prompt_len + len(per_row_ids[0]),
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

            if self.eos_token_id is not None and bool(
                (next_token == self.eos_token_id).all().item()
            ):
                break

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in per_row_ids
        ]

        self.logger.log_request(
            request_id=request_id,
            event="end",
            generated_tokens=len(per_row_ids[0]) * batch,
            final_text=texts[0] if batch == 1 else "\n---\n".join(texts),
        )
        return GenerationResult(texts=texts, generated_ids=per_row_ids)
