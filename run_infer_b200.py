#!/usr/bin/env python3
"""
LayerStream inference on a cloud GPU via the `Modal <https://modal.com>`_ API.

The GPU type is configurable at launch time — pass ``--gpu`` to the Modal entrypoint
(e.g. ``B200:1``, ``H100:2``, ``A100-80GB:1``, ``L40S:1``, ``L4:1``, ``T4:1``). The
default is ``B200:1`` to preserve the prior behavior. The flag is consumed by Modal
(not by ``src.run_infer``), so your inference flags are unchanged.

One-time setup::

    pip install modal
    modal setup

**HF_TOKEN**: if present in the local environment (e.g. exported from ``~/.bashrc``),
it is forwarded to the worker as an ephemeral Modal Secret. You can also reference
named secrets via ``MODAL_SECRET_NAMES=huggingface,...``.

Usage (repo root; everything after ``--`` is passed to ``python -m src.run_infer``)::

    modal run run_infer_b200.py --gpu H100:1 -- \
        --model Qwen/Qwen2.5-14B-Instruct --mode layered --prompt "Hello"

Alternative (single string, avoids Modal "unexpected extra arguments")::

    modal run run_infer_b200.py --gpu H100:1 \
        --infer-args "--model Qwen/Qwen2.5-14B-Instruct --mode layered --prompt 'Hello'"

Do **not** import ``torch`` at module level here; Modal evaluates this file when building the app.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

_APP_NAME = "layerstream-infer"
_REMOTE_ROOT = "/workspace/LayerStream"
_DEFAULT_GPU = "B200:1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _argv_before_double_dash() -> list[str]:
    """Tokens in ``sys.argv`` before a literal ``--`` separator (i.e. Modal-facing args)."""
    if "--" in sys.argv:
        return sys.argv[1 : sys.argv.index("--")]
    return sys.argv[1:]


def _infer_argv_from_modal_cli() -> list[str]:
    """Collect args for ``python -m src.run_infer`` — everything after a literal ``--``."""
    if "--" in sys.argv:
        i = sys.argv.index("--")
        return sys.argv[i + 1 :]
    return []


def _scan_gpu_from_argv() -> str:
    """
    Find ``--gpu <value>`` or ``--gpu=<value>`` in the Modal-facing argv.

    Modal's click parser evaluates the entrypoint signature *after* this module is
    imported, so we need to pre-read the flag to feed ``gpu=`` to ``@app.function``.
    """
    tokens = _argv_before_double_dash()
    for i, tok in enumerate(tokens):
        if tok == "--gpu" and i + 1 < len(tokens):
            return tokens[i + 1].strip()
        if tok.startswith("--gpu="):
            return tok.split("=", 1)[1].strip()
    return ""


def _effective_gpu() -> str:
    """Priority: ``--gpu`` in argv -> ``LAYERSTREAM_GPU`` env -> default."""
    return (
        _scan_gpu_from_argv()
        or os.environ.get("LAYERSTREAM_GPU", "").strip()
        or _DEFAULT_GPU
    )


def _modal_secrets() -> list[modal.Secret]:
    """
    Build the secret list passed to the Modal function.

    * If ``HF_TOKEN`` is set locally (e.g. from ``~/.bashrc``), wrap it in an ephemeral
      ``Secret.from_dict`` so the worker sees ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN``.
    * Also append any named secrets listed in ``MODAL_SECRET_NAMES`` (comma-separated).
    """
    secrets: list[modal.Secret] = []
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        secrets.append(
            modal.Secret.from_dict(
                {
                    "HF_TOKEN": hf_token,
                    "HUGGING_FACE_HUB_TOKEN": hf_token,
                }
            )
        )
    raw = os.environ.get("MODAL_SECRET_NAMES", "").strip()
    if raw:
        secrets.extend(
            modal.Secret.from_name(n.strip())
            for n in raw.split(",")
            if n.strip()
        )
    return secrets


_LAYERSTREAM_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "huggingface-hub>=0.22.0",
        "datasets>=2.16.0",
        "pynvml>=11.5.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(
        str(_repo_root()),
        remote_path=_REMOTE_ROOT,
        ignore=[".git", "__pycache__", ".cursor", "logs"],
    )
)

app = modal.App(_APP_NAME)

_GPU_SPEC = _effective_gpu()


@app.function(
    image=_LAYERSTREAM_IMAGE,
    gpu=_GPU_SPEC,
    timeout=86400,
    retries=0,
    secrets=_modal_secrets(),
)
def run_infer_remote(infer_argv: list[str]) -> int:
    """Run ``python -m src.run_infer`` in the mounted repo on a GPU worker."""
    import sys as _sys

    root = Path(_REMOTE_ROOT)
    if not (root / "src" / "run_infer.py").is_file():
        print(f"ERROR: expected LayerStream at {root}", file=_sys.stderr)
        return 2
    return int(
        subprocess.call(
            [_sys.executable, "-m", "src.run_infer", *infer_argv],
            cwd=str(root),
        )
    )


def _resolve_infer_argv(infer_args: str) -> list[str]:
    """Priority: explicit Modal ``--infer-args`` param -> env -> ``--`` passthrough."""
    s = infer_args.strip()
    if s:
        return shlex.split(s)
    env = os.environ.get("LAYERSTREAM_INFER_ARGS", "").strip()
    if env:
        return shlex.split(env)
    return _infer_argv_from_modal_cli()


@app.local_entrypoint()
def main(gpu: str = "", infer_args: str = "") -> None:
    # ``gpu`` is parsed again here purely so Click does not reject ``--gpu``; the value
    # that actually governs the remote worker's hardware is resolved at import time
    # (see ``_effective_gpu`` / ``_GPU_SPEC``). Warn if they somehow disagree.
    requested = gpu.strip() or os.environ.get("LAYERSTREAM_GPU", "").strip() or _DEFAULT_GPU
    if requested != _GPU_SPEC:
        print(
            f"[layerstream] warning: requested gpu={requested!r} but decorator bound "
            f"gpu={_GPU_SPEC!r}; pass --gpu before '--' or export LAYERSTREAM_GPU.",
            file=sys.stderr,
        )
    infer_argv = _resolve_infer_argv(infer_args)
    if not infer_argv:
        print(
            "No arguments passed through to src.run_infer.\n"
            "Examples:\n"
            "  modal run run_infer_b200.py --gpu H100:1 -- "
            '--model Qwen/Qwen2.5-7B-Instruct --mode layered --prompt "Hello"\n'
            "  modal run run_infer_b200.py --gpu A100-80GB:1 --infer-args "
            '"--model Qwen/Qwen2.5-7B-Instruct --mode layered --prompt \'Hello\'"\n',
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"[layerstream] launching on gpu={_GPU_SPEC}", file=sys.stderr)
    code = int(run_infer_remote.remote(infer_argv))
    raise SystemExit(code)


if __name__ == "__main__" and not os.environ.get("MODAL_TASK_ID"):
    print(
        "This file is a Modal app. Run on any Modal-supported GPU with:\n\n"
        "  modal run run_infer_b200.py --gpu <type[:count]> -- "
        "--model <hf_model_id> --mode layered ...\n\n"
        "Install: pip install modal && modal setup\n"
        "HF_TOKEN from your shell is forwarded automatically; named secrets via "
        "MODAL_SECRET_NAMES=huggingface\n",
        file=sys.stderr,
    )
    raise SystemExit(2)
