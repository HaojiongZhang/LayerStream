from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] missing {path}")
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return pd.DataFrame(rows)


def save_line(df: pd.DataFrame, x: str, ys: list[str], title: str, ylabel: str, out: Path):
    cols = [c for c in ys if c in df.columns]
    if df.empty or not cols or x not in df.columns:
        print(f"[skip] {title}")
        return

    plt.figure(figsize=(11, 5))
    for c in cols:
        plt.plot(df[x], df[c], label=c)

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[wrote] {out}")


def save_bar(df: pd.DataFrame, x: str, ys: list[str], title: str, ylabel: str, out: Path):
    cols = [c for c in ys if c in df.columns]
    if df.empty or not cols or x not in df.columns:
        print(f"[skip] {title}")
        return

    plot_df = df[[x] + cols].set_index(x)
    plot_df.plot(kind="bar", figsize=(12, 5))

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[wrote] {out}")


def save_heatmap(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    out: Path,
):
    if df.empty or not {index, columns, values}.issubset(df.columns):
        print(f"[skip] {title}")
        return

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="mean",
    )

    if pivot.empty:
        print(f"[skip] {title}")
        return

    plt.figure(figsize=(12, 7))
    plt.imshow(pivot, aspect="auto", interpolation="nearest")
    plt.colorbar(label=values)
    plt.title(title)
    plt.xlabel(columns)
    plt.ylabel(index)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[wrote] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_jsonl(log_dir / "tokens.jsonl")
    layers = load_jsonl(log_dir / "layers.jsonl")
    memory = load_jsonl(log_dir / "memory.jsonl")
    samples = load_jsonl(log_dir / "profile_samples.jsonl")
    requests = load_jsonl(log_dir / "requests.jsonl")

    # Normalize bytes -> MB for easier plotting.
    for df in [tokens, layers, memory, samples]:
        if not df.empty:
            for col in ["kv_gpu_bytes", "gpu_kv_bytes"]:
                if col in df.columns:
                    df[col.replace("bytes", "mb")] = df[col] / (1024 ** 2)
            for col in ["kv_cpu_bytes", "cpu_kv_bytes"]:
                if col in df.columns:
                    df[col.replace("bytes", "mb")] = df[col] / (1024 ** 2)

    # Profile sampled time-series.
    if not samples.empty and "ts" in samples.columns:
        samples = samples.sort_values("ts").copy()
        samples["t_s"] = samples["ts"] - samples["ts"].iloc[0]

        save_line(
            samples,
            "t_s",
            ["gpu_alloc_mb", "gpu_reserved_mb"],
            "Sampled GPU memory over time",
            "MB",
            out_dir / "sampled_gpu_memory.png",
        )

        save_line(
            samples,
            "t_s",
            ["kv_gpu_mb", "kv_cpu_mb"],
            "Sampled KV cache memory over time",
            "MB",
            out_dir / "sampled_kv_memory.png",
        )

        save_line(
            samples,
            "t_s",
            ["gpu_util_pct", "mem_bus_util_pct"],
            "GPU utilization over time",
            "%",
            out_dir / "gpu_utilization.png",
        )

    # Per-token timings.
    if not tokens.empty:
        tokens = tokens.sort_values(["request_id", "token_idx"]).copy()

        save_line(
            tokens,
            "token_idx",
            ["token_latency_ms", "copy_total_ms", "compute_total_ms", "wait_total_ms"],
            "Per-token latency breakdown",
            "ms",
            out_dir / "token_latency_breakdown.png",
        )

        save_line(
            tokens,
            "token_idx",
            ["gpu_allocated_mb", "gpu_reserved_mb"],
            "Per-token GPU memory",
            "MB",
            out_dir / "token_gpu_memory.png",
        )

        save_line(
            tokens,
            "token_idx",
            ["kv_gpu_mb", "kv_cpu_mb"],
            "Per-token KV cache memory",
            "MB",
            out_dir / "token_kv_memory.png",
        )

        if "token_latency_ms" in tokens.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(tokens["token_latency_ms"].dropna(), bins=40)
            plt.title("Token latency distribution")
            plt.xlabel("Latency (ms)")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "token_latency_hist.png", dpi=180)
            plt.close()

    # Per-layer heatmaps.
    if not layers.empty:
        layers = layers.sort_values(["request_id", "token_idx", "layer_idx"]).copy()

        compute = layers[layers.get("event", "") == "compute"].copy()
        prefetch = layers[layers.get("event", "") == "prefetch"].copy()

        save_heatmap(
            compute,
            index="layer_idx",
            columns="token_idx",
            values="compute_ms",
            title="Layer compute time heatmap",
            out=out_dir / "layer_compute_heatmap.png",
        )

        save_heatmap(
            compute,
            index="layer_idx",
            columns="token_idx",
            values="wait_ms",
            title="Layer wait time heatmap",
            out=out_dir / "layer_wait_heatmap.png",
        )

        save_heatmap(
            prefetch,
            index="layer_idx",
            columns="token_idx",
            values="copy_ms",
            title="Layer prefetch copy time heatmap",
            out=out_dir / "layer_prefetch_copy_heatmap.png",
        )

        if not compute.empty:
            by_layer = compute.groupby("layer_idx", as_index=False).agg(
                compute_ms_mean=("compute_ms", "mean"),
                compute_ms_p95=("compute_ms", lambda x: x.quantile(0.95)),
                wait_ms_mean=("wait_ms", "mean"),
                wait_ms_p95=("wait_ms", lambda x: x.quantile(0.95)),
            )

            save_line(
                by_layer,
                "layer_idx",
                ["compute_ms_mean", "compute_ms_p95", "wait_ms_mean", "wait_ms_p95"],
                "Average per-layer compute/wait time",
                "ms",
                out_dir / "layer_compute_wait_by_layer.png",
            )

        if not prefetch.empty:
            by_layer_copy = prefetch.groupby("layer_idx", as_index=False).agg(
                copy_ms_mean=("copy_ms", "mean"),
                copy_ms_p95=("copy_ms", lambda x: x.quantile(0.95)),
                weight_mb_mean=("weight_bytes", lambda x: x.mean() / (1024 ** 2))
                if "weight_bytes" in prefetch.columns else ("copy_ms", "mean"),
            )

            save_line(
                by_layer_copy,
                "layer_idx",
                ["copy_ms_mean", "copy_ms_p95"],
                "Average per-layer prefetch copy time",
                "ms",
                out_dir / "layer_prefetch_copy_by_layer.png",
            )

            if "buffer_slot" in prefetch.columns:
                slot_counts = prefetch.groupby(["token_idx", "buffer_slot"]).size().reset_index(name="count")
                save_bar(
                    slot_counts,
                    "token_idx",
                    ["count"],
                    "Prefetch events per token/slot",
                    "Count",
                    out_dir / "prefetch_slot_counts.png",
                )

    # Memory log.
    if not memory.empty:
        memory = memory.sort_values(["request_id", "token_idx"]).copy()
        save_line(
            memory,
            "token_idx",
            ["gpu_allocated_mb", "gpu_reserved_mb", "max_allocated_mb", "max_reserved_mb"],
            "Memory log over tokens",
            "MB",
            out_dir / "memory_log.png",
        )

    # Simple text summary.
    summary_path = log_dir / "profile_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    with (out_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("# LayerStream Visualization Report\n\n")
        f.write(f"Log dir: `{log_dir}`\n\n")

        if summary:
            f.write("## Profile summary\n\n")
            for k, v in summary.items():
                f.write(f"- `{k}`: `{v}`\n")
            f.write("\n")

        if not tokens.empty:
            f.write("## Token stats\n\n")
            f.write(f"- Num token rows: `{len(tokens)}`\n")
            if "token_latency_ms" in tokens.columns:
                f.write(f"- Mean token latency: `{tokens['token_latency_ms'].mean():.3f}` ms\n")
                f.write(f"- P95 token latency: `{tokens['token_latency_ms'].quantile(0.95):.3f}` ms\n")
            if "copy_total_ms" in tokens.columns:
                f.write(f"- Mean copy total: `{tokens['copy_total_ms'].mean():.3f}` ms\n")
            if "compute_total_ms" in tokens.columns:
                f.write(f"- Mean compute total: `{tokens['compute_total_ms'].mean():.3f}` ms\n")
            if "wait_total_ms" in tokens.columns:
                f.write(f"- Mean wait total: `{tokens['wait_total_ms'].mean():.3f}` ms\n")
            f.write("\n")

        if not layers.empty:
            f.write("## Layer stats\n\n")
            f.write(f"- Num layer rows: `{len(layers)}`\n")
            if "event" in layers.columns:
                f.write(f"- Events: `{layers['event'].value_counts().to_dict()}`\n")
            f.write("\n")

        if not requests.empty:
            f.write("## Requests\n\n")
            f.write(f"- Num request rows: `{len(requests)}`\n")

    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()