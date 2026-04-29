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
    
    @torch.inference_mode()
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

    @torch.inference_mode()
    def generate_batch(
        self,
        request_ids: list[int],
        prompts: list[str],
        max_new_tokens: int = 64,
    ) -> list[GenerationResult]:
        """Batched prefill + decode for a list of prompts.

        Constraints:
        - All prompts must tokenize to the **same length**. The forward engine's
          positional encoding (RoPE / cache_position) is computed from a single
          ``kv_manager.current_seq_len`` shared across the batch, so left-padded
          rows would receive shifted positions and produce wrong outputs.
          Callers (e.g. ``run_ruler_eval``) should group same-length samples per
          batch and fall back to ``generate()`` when lengths differ.
        - Caller must ``kv_manager.reset()`` before each batch — KV state is
          shared across the whole batch's generation, then dropped.

        Args:
            request_ids: per-row request id, used for log_request start/end.
            prompts: list of prompt strings; must be non-empty and uniform-len.
            max_new_tokens: per-row decode cap. Decode stops early if every row
                emits EOS.

        Returns:
            list[GenerationResult] in the same order as ``prompts``.
        """
        if len(prompts) != len(request_ids):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(request_ids)={len(request_ids)}"
            )
        if not prompts:
            return []

        # Tokenize each prompt independently first to validate uniform length —
        # the engine's shared positional encoding can't handle mixed lengths.
        per_row_ids = [
            self.tokenizer(p, return_tensors="pt")["input_ids"][0] for p in prompts
        ]
        prompt_lens = [int(t.shape[0]) for t in per_row_ids]
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "generate_batch requires all prompts to tokenize to the same "
                f"length; got {prompt_lens}. Group same-length prompts per batch "
                "or fall back to generate() for mixed lengths."
            )

        prompt_len = prompt_lens[0]
        batch_size = len(prompts)

        # Stack into (B, T) — no padding needed since all rows are equal len.
        input_ids = torch.stack(per_row_ids, dim=0).to("cuda")  # (B, T)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

        for rid, prompt in zip(request_ids, prompts):
            self.logger.log_request(
                request_id=rid,
                event="start",
                prompt=prompt,
                prompt_len=prompt_len,
            )

        eos_id = self.eos_token_id
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        # Use the first request_id for batch-level token/memory logs. Per-row
        # request_ids are only used for log_request start/end above + below.
        batch_log_id = request_ids[0]

        cur_input_ids = input_ids
        cur_attn = attention_mask

        for step in range(max_new_tokens):
            token_timer = WallTimer()
            token_timer.start()

            out = self.engine.forward_token(
                request_id=batch_log_id,
                token_idx=step,
                input_ids=cur_input_ids,
                attention_mask=cur_attn,
                position_ids=None,
            )

            next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)  # (B,)
            next_tokens_list = next_tokens.tolist()

            for b, tok_id in enumerate(next_tokens_list):
                if finished[b]:
                    continue
                generated[b].append(int(tok_id))
                if eos_id is not None and tok_id == eos_id:
                    finished[b] = True

            cur_input_ids = next_tokens.view(batch_size, 1).to(input_ids.dtype)
            cur_attn = torch.cat(
                [
                    cur_attn,
                    torch.ones((batch_size, 1), device=cur_attn.device, dtype=cur_attn.dtype),
                ],
                dim=1,
            )

            token_timer.stop()
            allocated_mb, reserved_mb = self._gpu_mem_mb()

            self.logger.log_token(
                request_id=batch_log_id,
                token_idx=step,
                context_len=prompt_len + step + 1,
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
                request_id=batch_log_id,
                token_idx=step,
                gpu_allocated_mb=allocated_mb,
                gpu_reserved_mb=reserved_mb,
                max_allocated_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
                max_reserved_mb=torch.cuda.max_memory_reserved() / (1024 ** 2),
            )

            if all(finished):
                break

        results: list[GenerationResult] = []
        for rid, gen_ids in zip(request_ids, generated):
            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            self.logger.log_request(
                request_id=rid,
                event="end",
                generated_tokens=len(gen_ids),
                final_text=text,
            )
            results.append(GenerationResult(text=text, generated_ids=gen_ids))

        return results