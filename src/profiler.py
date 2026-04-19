"""
Lightweight sampling profiler for LayerStream.

Captures, while generation runs:
  * GPU memory (allocated / reserved / torch peak stats)
  * KV-cache bytes on GPU and CPU (via the KVCacheManager)
  * GPU compute + memory-bus utilization (via NVML, if available)
  * Wall-clock throughput (tokens / s) and per-token latency distribution
    (mean, p50, p95, p99) derived from logs/tokens.jsonl

A background daemon thread polls memory + utilization at `sample_hz` and
streams each sample to logs/profile_samples.jsonl. The final summary is
written to logs/profile_summary.json and pretty-printed to stdout.

Activated from run_infer.py with `--profile`.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .kv_manager import KVCacheManager


def _try_init_nvml() -> Optional[Any]:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    return pynvml


class Profiler:
    def __init__(
        self,
        log_dir: Path,
        kv_manager: KVCacheManager,
        sample_hz: float = 20.0,
        device_index: int = 0,
        request_id: int = 1,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.kv_manager = kv_manager
        self.sample_interval_s = 1.0 / max(sample_hz, 1.0)
        self.device_index = device_index
        self.request_id = request_id

        self._nvml = _try_init_nvml()
        self._nvml_handle = None
        if self._nvml is not None:
            try:
                self._nvml_handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._nvml = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples_path = self.log_dir / "profile_samples.jsonl"
        self._summary_path = self.log_dir / "profile_summary.json"

        self._t0_perf: Optional[float] = None
        self._t1_perf: Optional[float] = None
        self._t0_wall: Optional[float] = None

        self._peak_gpu_alloc_mb = 0.0
        self._peak_gpu_reserved_mb = 0.0
        self._peak_kv_gpu_bytes = 0
        self._peak_kv_cpu_bytes = 0
        self._gpu_util_samples: list[float] = []
        self._mem_util_samples: list[float] = []
        self._num_samples = 0

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_index)
        self._stop_event.clear()
        self._t0_wall = time.time()
        self._t0_perf = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="LayerStreamProfiler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._t1_perf = time.perf_counter()
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self) -> None:
        with self._samples_path.open("w", encoding="utf-8") as f:
            while True:
                sample = self._sample_once()
                self._update_peaks(sample)
                f.write(json.dumps(sample) + "\n")
                f.flush()
                if self._stop_event.wait(self.sample_interval_s):
                    break

    def _sample_once(self) -> dict[str, Any]:
        # memory_allocated / memory_reserved read atomic counters; safe from
        # another thread and don't trigger kernel launches.
        alloc_mb = torch.cuda.memory_allocated(self.device_index) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)
        kv_gpu = self.kv_manager.total_gpu_kv_bytes()
        kv_cpu = self.kv_manager.total_cpu_kv_bytes()

        gpu_util: Optional[float] = None
        mem_util: Optional[float] = None
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(u.gpu)
                mem_util = float(u.memory)
                self._gpu_util_samples.append(gpu_util)
                self._mem_util_samples.append(mem_util)
            except Exception:
                pass

        self._num_samples += 1
        return {
            "ts": time.time(),
            "gpu_alloc_mb": alloc_mb,
            "gpu_reserved_mb": reserved_mb,
            "kv_gpu_bytes": kv_gpu,
            "kv_cpu_bytes": kv_cpu,
            "kv_gpu_mb": kv_gpu / (1024 ** 2),
            "kv_cpu_mb": kv_cpu / (1024 ** 2),
            "gpu_util_pct": gpu_util,
            "mem_bus_util_pct": mem_util,
        }

    def _update_peaks(self, s: dict[str, Any]) -> None:
        if s["gpu_alloc_mb"] > self._peak_gpu_alloc_mb:
            self._peak_gpu_alloc_mb = s["gpu_alloc_mb"]
        if s["gpu_reserved_mb"] > self._peak_gpu_reserved_mb:
            self._peak_gpu_reserved_mb = s["gpu_reserved_mb"]
        if s["kv_gpu_bytes"] > self._peak_kv_gpu_bytes:
            self._peak_kv_gpu_bytes = s["kv_gpu_bytes"]
        if s["kv_cpu_bytes"] > self._peak_kv_cpu_bytes:
            self._peak_kv_cpu_bytes = s["kv_cpu_bytes"]

    def _read_token_latencies_ms(self) -> list[float]:
        """Pull per-token latencies from tokens.jsonl for this run only."""
        tokens_file = self.log_dir / "tokens.jsonl"
        if not tokens_file.exists() or self._t0_wall is None:
            return []

        latencies: list[float] = []
        with tokens_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("request_id") != self.request_id:
                    continue
                if rec.get("ts", 0.0) < self._t0_wall:
                    continue
                lat = rec.get("token_latency_ms")
                if lat is not None:
                    latencies.append(float(lat))
        return latencies

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        if len(s) == 1:
            return s[0]
        k = (p / 100.0) * (len(s) - 1)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    def summary(self, prompt_len: int, generated_tokens: int) -> dict[str, Any]:
        wall_s = 0.0
        if self._t0_perf is not None and self._t1_perf is not None:
            wall_s = self._t1_perf - self._t0_perf

        latencies = self._read_token_latencies_ms()

        torch_peak_alloc_mb = 0.0
        torch_peak_reserved_mb = 0.0
        if torch.cuda.is_available():
            torch_peak_alloc_mb = (
                torch.cuda.max_memory_allocated(self.device_index) / (1024 ** 2)
            )
            torch_peak_reserved_mb = (
                torch.cuda.max_memory_reserved(self.device_index) / (1024 ** 2)
            )

        return {
            "wall_s": wall_s,
            "prompt_len": prompt_len,
            "generated_tokens": generated_tokens,
            "tokens_per_s": (generated_tokens / wall_s) if wall_s > 0 else 0.0,
            "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
            "latency_ms_p50": self._percentile(latencies, 50),
            "latency_ms_p95": self._percentile(latencies, 95),
            "latency_ms_p99": self._percentile(latencies, 99),
            "peak_gpu_alloc_mb_sampled": self._peak_gpu_alloc_mb,
            "peak_gpu_reserved_mb_sampled": self._peak_gpu_reserved_mb,
            "peak_gpu_alloc_mb_torch": torch_peak_alloc_mb,
            "peak_gpu_reserved_mb_torch": torch_peak_reserved_mb,
            "peak_kv_gpu_mb": self._peak_kv_gpu_bytes / (1024 ** 2),
            "peak_kv_cpu_mb": self._peak_kv_cpu_bytes / (1024 ** 2),
            "gpu_util_pct_mean": (
                statistics.mean(self._gpu_util_samples)
                if self._gpu_util_samples else None
            ),
            "gpu_util_pct_peak": (
                max(self._gpu_util_samples) if self._gpu_util_samples else None
            ),
            "mem_bus_util_pct_mean": (
                statistics.mean(self._mem_util_samples)
                if self._mem_util_samples else None
            ),
            "num_samples": self._num_samples,
            "sample_hz": 1.0 / self.sample_interval_s,
            "nvml_available": self._nvml is not None,
            "num_tokens_with_latency": len(latencies),
        }

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self._summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, summary: dict[str, Any]) -> None:
        print("\n=== LayerStream Profile Summary ===")
        print(f"Wall time:           {summary['wall_s']:.3f} s")
        print(f"Prompt tokens:       {summary['prompt_len']}")
        print(f"Generated tokens:    {summary['generated_tokens']}")
        print(f"Throughput:          {summary['tokens_per_s']:.2f} tok/s")
        print(
            f"Latency mean:        {summary['latency_ms_mean']:.2f} ms  "
            f"(from {summary['num_tokens_with_latency']} tokens)"
        )
        print(
            f"Latency p50/p95/p99: "
            f"{summary['latency_ms_p50']:.2f} / "
            f"{summary['latency_ms_p95']:.2f} / "
            f"{summary['latency_ms_p99']:.2f} ms"
        )
        print(
            f"Peak GPU alloc:      "
            f"{summary['peak_gpu_alloc_mb_torch']:.1f} MB (torch)  "
            f"{summary['peak_gpu_alloc_mb_sampled']:.1f} MB (sampled)"
        )
        print(
            f"Peak GPU reserved:   "
            f"{summary['peak_gpu_reserved_mb_torch']:.1f} MB (torch)  "
            f"{summary['peak_gpu_reserved_mb_sampled']:.1f} MB (sampled)"
        )
        print(f"Peak KV on GPU:      {summary['peak_kv_gpu_mb']:.1f} MB")
        print(f"Peak KV on CPU:      {summary['peak_kv_cpu_mb']:.1f} MB")
        if summary["nvml_available"] and summary["gpu_util_pct_mean"] is not None:
            print(
                f"GPU util mean/peak:  "
                f"{summary['gpu_util_pct_mean']:.1f}% / "
                f"{summary['gpu_util_pct_peak']:.1f}%"
            )
            print(f"Mem bus util mean:   {summary['mem_bus_util_pct_mean']:.1f}%")
        else:
            print("GPU util:            (install pynvml for compute utilization)")
        print(
            f"Samples captured:    {summary['num_samples']} "
            f"@ {summary['sample_hz']:.1f} Hz"
        )
        print(f"Samples file:        {self._samples_path}")
        print(f"Summary file:        {self._summary_path}")
        print("===================================\n")
