from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MEM_RE = re.compile(
    r"^\[mem\]\s+"
    r"(?P<tag>.*?):\s+"
    r"alloc=(?P<alloc>[0-9.]+)MB\s+"
    r"reserved=(?P<reserved>[0-9.]+)MB\s+"
    r"max_alloc=(?P<max_alloc>[0-9.]+)MB\s+"
    r"max_reserved=(?P<max_reserved>[0-9.]+)MB\s+"
    r"kv_gpu=(?P<kv_gpu>[0-9.]+)MB\s+"
    r"kv_cpu=(?P<kv_cpu>[0-9.]+)MB"
)

TOK_LAYER_RE = re.compile(r"tok=(?P<tok>\d+)(?:\s+layer=(?P<layer>\d+))?")


def parse_mem_trace(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f):
            m = MEM_RE.match(line.strip())
            if not m:
                continue

            tag = m.group("tag")
            tl = TOK_LAYER_RE.search(tag)

            rows.append(
                {
                    "step": len(rows),
                    "line_idx": line_idx,
                    "tag": tag,
                    "token_idx": int(tl.group("tok")) if tl else None,
                    "layer_idx": int(tl.group("layer")) if tl and tl.group("layer") else None,
                    "alloc_mb": float(m.group("alloc")),
                    "reserved_mb": float(m.group("reserved")),
                    "max_alloc_mb": float(m.group("max_alloc")),
                    "max_reserved_mb": float(m.group("max_reserved")),
                    "kv_gpu_mb": float(m.group("kv_gpu")),
                    "kv_cpu_mb": float(m.group("kv_cpu")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["non_kv_alloc_mb"] = df["alloc_mb"] - df["kv_gpu_mb"]
    df["delta_alloc_mb"] = df["alloc_mb"].diff()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent / "mem_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_mem_trace(log_path)
    if df.empty:
        raise SystemExit(f"No [mem] lines found in {log_path}")

    df.to_csv(out_dir / "mem_trace_parsed.csv", index=False)

    # One combined plot.
    plt.figure(figsize=(14, 6))
    plt.plot(df["step"], df["alloc_mb"], label="total_alloc_mb")
    plt.plot(df["step"], df["reserved_mb"], label="reserved_mb")
    plt.plot(df["step"], df["kv_gpu_mb"], label="kv_gpu_mb")
    plt.plot(df["step"], df["kv_cpu_mb"], label="kv_cpu_mb")
    plt.plot(df["step"], df["non_kv_alloc_mb"], label="non_kv_alloc_mb")

    # Mark token boundaries if multiple tokens are present.
    if "token_idx" in df.columns:
        boundaries = df.groupby("token_idx")["step"].min().dropna()
        for tok, step in boundaries.items():
            plt.axvline(step, linestyle="--", linewidth=0.8, alpha=0.35)
            plt.text(step, plt.ylim()[1] * 0.95, f"tok {int(tok)}", rotation=90, va="top", fontsize=8)

    plt.title("Memory trace across debug steps")
    plt.xlabel("Debug step")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = out_dir / "memory_trace.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    # One CSV for biggest jumps.
    jump_cols = [
        "step",
        "token_idx",
        "layer_idx",
        "tag",
        "alloc_mb",
        "delta_alloc_mb",
        "reserved_mb",
        "kv_gpu_mb",
        "kv_cpu_mb",
        "non_kv_alloc_mb",
    ]
    jumps = df.sort_values("delta_alloc_mb", ascending=False).head(30)
    jumps[jump_cols].to_csv(out_dir / "largest_memory_jumps.csv", index=False)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {out_dir / 'largest_memory_jumps.csv'}")
    print(f"Wrote: {out_dir / 'mem_trace_parsed.csv'}")

    print("\nLargest allocation jumps:")
    print(jumps[jump_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()