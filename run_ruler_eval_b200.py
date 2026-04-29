#!/usr/bin/env python3
"""
RULER / NIAH evaluation on a cloud GPU via `Modal <https://modal.com>`_.

Wraps ``python -m src.run_ruler_eval`` so the same flags you use locally also
work remotely. The GPU type is configurable at launch — pass ``--gpu``
(e.g. ``B200:1``, ``H100:1``, ``A100-80GB:1``, ``L40S:1``). Default is ``B200:1``.

``run_ruler_eval`` supports ``--batch-size N`` for batched generation. Pass
through here unchanged in ``--eval-args``; e.g. ``--batch-size 4`` evaluates
four uniform-length prompts per ``generate_batch()`` call with a KV-cache
reset between batches. Mixed-length batches automatically fall back to
sequential ``generate()`` per row.

Modal parses its own CLI first. Pass eval flags in one of these ways:

1. **Double dash** (everything after ``--`` goes to ``src.run_ruler_eval``)::

       modal run run_ruler_eval_b200.py --gpu H100:1 -- \
           --model meta-llama/Llama-2-13b-chat-hf \
           --mode layered \
           --dataset-path ./ruler_data/niah_single_1_4096.jsonl \
           --task niah_single_1 \
           --kv-cache-bits 16 \
           --max-seq-len 8192 \
           --max-new-tokens 64 \
           --num-samples 20 \
           --profile \
           --log-dir logs/ruler_niah_single_1_8192_kv16

2. **Modal entrypoint arg** (single string; avoids Modal "unexpected extra arguments")::

       modal run run_ruler_eval_b200.py --gpu H100:1 \
           --eval-args "--model meta-llama/Llama-2-13b-chat-hf --mode layered \
                        --dataset-path ./ruler_data/niah_single_1_4096.jsonl \
                        --task niah_single_1 --kv-cache-bits 16 --max-seq-len 8192 \
                        --max-new-tokens 64 --num-samples 20 --profile \
                        --log-dir logs/ruler_niah_single_1_8192_kv16"

3. **Environment** (no extra CLI tokens after the script path)::

       export LAYERSTREAM_GPU=H100:1
       export LAYERSTREAM_EVAL_ARGS="--model meta-llama/Llama-2-13b-chat-hf --mode layered ..."
       modal run run_ruler_eval_b200.py

One-time::

    pip install modal && modal setup

Dataset note: ``--dataset-path`` is resolved relative to the repo root on the
worker. Place the JSONL (e.g. ``ruler_data/niah_single_1_4096.jsonl``) in the
repo before launching — ``add_local_dir`` uploads it with the rest of the tree.

Logs: by default ``--log-dir logs/...`` writes inside the worker's container
and is *not* synced back. To persist logs across runs, set
``LAYERSTREAM_LOGS_VOLUME=<volume-name>`` and the worker will mount that Modal
Volume at the repo's ``logs/`` directory.

**HF_TOKEN**: if present in the local environment, it is forwarded to the
worker as an ephemeral Modal Secret so gated models (Llama-2, etc.) work.
You can still reference named Modal Secrets via
``MODAL_SECRET_NAMES=huggingface,...``.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

_APP_NAME = "layerstream-ruler-eval"
_REMOTE_ROOT = "/workspace/LayerStream"
_DEFAULT_GPU = "B200:1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _argv_before_double_dash() -> list[str]:
    """Tokens in ``sys.argv`` before a literal ``--`` separator (Modal-facing args)."""
    if "--" in sys.argv:
        return sys.argv[1 : sys.argv.index("--")]
    return sys.argv[1:]


def _eval_argv_from_modal_cli() -> list[str]:
    """Args after a literal ``--`` in ``sys.argv`` (Modal forwards them for ``modal run app.py -- ...``)."""
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


_DEFAULT_LOGS_VOLUME = "layerstream-logs"
_DEFAULT_HF_VOLUME = "hf-cache"


def _modal_volumes() -> dict[str, modal.Volume]:
    """
    Always-on persistent volumes (override names via env if you want):

    * ``LAYERSTREAM_LOGS_VOLUME`` (default ``layerstream-logs``) -> mounted at
      ``<repo>/logs`` so ``--log-dir logs/...`` survives across worker
      invocations. Pull contents back with ``modal volume get <name> <path>``.
    * ``LAYERSTREAM_HF_CACHE_VOLUME`` (default ``hf-cache``) -> mounted at
      ``/root/.cache/huggingface`` so model weights persist between runs.

    Volumes are unconditional so the launcher and container always agree on
    the dependency count (a previous bug: optional volumes registered as
    extra container dep ids and tripped Modal's "Function has X dependencies
    but container got Y object ids" check).
    """
    logs_name = (
        os.environ.get("LAYERSTREAM_LOGS_VOLUME", "").strip() or _DEFAULT_LOGS_VOLUME
    )
    hf_name = (
        os.environ.get("LAYERSTREAM_HF_CACHE_VOLUME", "").strip() or _DEFAULT_HF_VOLUME
    )
    return {
        f"{_REMOTE_ROOT}/logs": modal.Volume.from_name(
            logs_name, create_if_missing=True
        ),
        "/root/.cache/huggingface": modal.Volume.from_name(
            hf_name, create_if_missing=True
        ),
    }


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
    # Forward the volume-name env vars into the image so module-level code
    # (``_modal_volumes``) resolves to the *same* set of volumes on the local
    # launcher and inside the container. Without this, the launcher would
    # register N volumes while the container re-imports the module with the
    # vars unset and registers 0, triggering Modal's "Function has X
    # dependencies but container got Y object ids" error.
    .env(
        {
            "LAYERSTREAM_HF_CACHE_VOLUME": os.environ.get(
                "LAYERSTREAM_HF_CACHE_VOLUME", ""
            ),
            "LAYERSTREAM_LOGS_VOLUME": os.environ.get(
                "LAYERSTREAM_LOGS_VOLUME", ""
            ),
        }
    )
    .add_local_dir(
        str(_repo_root()),
        remote_path=_REMOTE_ROOT,
        ignore=[".git", "__pycache__", ".cursor", "logs", "RULER"],
    )
)

app = modal.App(_APP_NAME)

_GPU_SPEC = _effective_gpu()
_VOLUMES = _modal_volumes()


@app.function(
    image=_LAYERSTREAM_IMAGE,
    gpu=_GPU_SPEC,
    timeout=86400,
    retries=0,
    secrets=_modal_secrets(),
    volumes=_VOLUMES,
)
def run_ruler_eval_remote(eval_argv: list[str]) -> int:
    """Run ``python -m src.run_ruler_eval`` on a GPU worker (GPU type bound at import time)."""
    import sys as _sys

    root = Path(_REMOTE_ROOT)
    if not (root / "src" / "run_ruler_eval.py").is_file():
        print(f"ERROR: expected LayerStream at {root}", file=_sys.stderr)
        return 2

    # The logs volume is mounted at <root>/logs; make sure the dir exists
    # so ``--log-dir logs/<subdir>`` can mkdir into it.
    (root / "logs").mkdir(parents=True, exist_ok=True)

    rc = int(
        subprocess.call(
            [_sys.executable, "-m", "src.run_ruler_eval", *eval_argv],
            cwd=str(root),
        )
    )

    # Explicitly flush volume writes back to Modal. Auto-commit on function
    # exit can miss late writes from the subprocess we just spawned, leaving
    # the volume empty even though the eval succeeded.
    try:
        for vol in _VOLUMES.values():
            vol.commit()
    except Exception as e:  # pragma: no cover - best-effort flush
        print(f"[layerstream] volume.commit() failed: {e!r}", file=_sys.stderr)

    return rc


def _resolve_eval_argv(eval_args: str) -> list[str]:
    """Priority: explicit Modal param -> env -> ``--`` passthrough."""
    s = eval_args.strip()
    if s:
        return shlex.split(s)
    env = os.environ.get("LAYERSTREAM_EVAL_ARGS", "").strip()
    if env:
        return shlex.split(env)
    return _eval_argv_from_modal_cli()


@app.local_entrypoint()
def main(gpu: str = "", eval_args: str = "") -> None:
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
    eval_argv = _resolve_eval_argv(eval_args)
    if not eval_argv:
        print(
            "No arguments for src.run_ruler_eval. Use one of:\n\n"
            "  modal run run_ruler_eval_b200.py --gpu H100:1 -- \\\n"
            "      --model meta-llama/Llama-2-13b-chat-hf \\\n"
            "      --mode layered \\\n"
            "      --dataset-path ./ruler_data/niah_single_1_4096.jsonl \\\n"
            "      --task niah_single_1 \\\n"
            "      --kv-cache-bits 16 --max-seq-len 8192 --max-new-tokens 64 \\\n"
            "      --num-samples 20 --profile \\\n"
            "      --log-dir logs/ruler_niah_single_1_8192_kv16\n\n"
            "  modal run run_ruler_eval_b200.py --gpu H100:1 --eval-args \\\n"
            '      "--model meta-llama/Llama-2-13b-chat-hf --mode layered '
            "--dataset-path ./ruler_data/niah_single_1_4096.jsonl --task niah_single_1 "
            "--kv-cache-bits 16 --max-seq-len 4096 --max-new-tokens 64 --num-samples 1 "
            '--profile --log-dir logs/ruler_niah_single_1_4096_kv16"\n\n'
            "  export LAYERSTREAM_GPU=H100:1\n"
            "  export LAYERSTREAM_EVAL_ARGS='--model meta-llama/Llama-2-13b-chat-hf ...'\n"
            "  modal run run_ruler_eval_b200.py\n",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"[layerstream] launching on gpu={_GPU_SPEC}", file=sys.stderr)
    code = int(run_ruler_eval_remote.remote(eval_argv))

    # Auto-sync the run's logs from the Modal Volume back to local disk so
    # ``logs/<subdir>/...`` is visible on the launching machine without a
    # manual ``modal volume get``. Failures here are non-fatal — they don't
    # change the eval's exit code.
    try:
        _sync_logs_from_volume(eval_argv)
    except Exception as e:
        print(f"[layerstream] log sync failed: {e!r}", file=sys.stderr)

    raise SystemExit(code)


def _extract_log_dir(eval_argv: list[str]) -> str | None:
    """Return the value of ``--log-dir`` (or ``--log-dir=...``) in argv, else None."""
    for i, tok in enumerate(eval_argv):
        if tok == "--log-dir" and i + 1 < len(eval_argv):
            return eval_argv[i + 1].strip()
        if tok.startswith("--log-dir="):
            return tok.split("=", 1)[1].strip()
    return None


def _sync_logs_from_volume(eval_argv: list[str]) -> None:
    """
    Pull the run's logs from the layerstream-logs Modal Volume to local disk.

    The volume is mounted on the worker at ``<repo>/logs``, so a ``--log-dir
    logs/<subdir>`` flag on the eval results in writes to ``<subdir>`` on the
    volume root. We download just that subdir into ``<repo_root>/logs/<subdir>``
    locally so the layout matches what the eval saw remotely.
    """
    log_dir = _extract_log_dir(eval_argv)
    if not log_dir:
        # nothing to do — eval didn't pass --log-dir
        return

    p = Path(log_dir)
    parts = p.parts
    # accept ``logs/<subdir>``, ``./logs/<subdir>``, or just ``<subdir>``
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "logs":
        parts = parts[1:]
    if not parts:
        # whole volume; pull root recursively
        remote_path = "/"
        local_dest = _repo_root() / "logs"
    else:
        remote_path = "/".join(parts)
        local_dest = _repo_root() / "logs" / Path(*parts)

    logs_volume_name = (
        os.environ.get("LAYERSTREAM_LOGS_VOLUME", "").strip() or _DEFAULT_LOGS_VOLUME
    )

    # Wipe and recreate so re-runs of the same --log-dir aren't merged with
    # stale files. ``modal volume get`` errors if the local dest already
    # exists for a directory pull.
    if local_dest.exists():
        import shutil

        shutil.rmtree(local_dest)
    local_dest.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[layerstream] syncing {logs_volume_name}:{remote_path} -> {local_dest}",
        file=sys.stderr,
    )
    rc = subprocess.call(
        [
            "modal",
            "volume",
            "get",
            logs_volume_name,
            remote_path,
            str(local_dest),
        ],
    )
    if rc != 0:
        print(
            f"[layerstream] modal volume get exited with rc={rc}; "
            f"run manually: modal volume get {logs_volume_name} {remote_path} {local_dest}",
            file=sys.stderr,
        )


if __name__ == "__main__" and not os.environ.get("MODAL_TASK_ID"):
    print(
        "This file is a Modal app. Examples:\n\n"
        "  modal run run_ruler_eval_b200.py --gpu H100:1 -- \\\n"
        "      --model meta-llama/Llama-2-13b-chat-hf --mode layered \\\n"
        "      --dataset-path ./ruler_data/niah_single_1_4096.jsonl \\\n"
        "      --task niah_single_1 --kv-cache-bits 16 --max-seq-len 8192 \\\n"
        "      --max-new-tokens 64 --num-samples 20 --profile \\\n"
        "      --log-dir logs/ruler_niah_single_1_8192_kv16\n",
        file=sys.stderr,
    )
    raise SystemExit(2)
