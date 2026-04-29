from __future__ import annotations

import os
import random
from typing import Any, Iterator

from .mcq_utils import format_mcq_prompt


def _hf_token_for_hub() -> str | bool:
    """
    Token for ``datasets.load_dataset`` on gated repos.

    Uses ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` if set; otherwise ``True``
    (use credentials from ``huggingface-cli login`` cache when present).
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return True


def _get(row: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def iter_gpqa_mc(
    split: str = "train",
    subset: str = "gpqa_diamond",
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[tuple[str, str, list[str], str]]:
    """
    Yield (example_id, gold_letter, shuffled_choices, question).

    Uses the **gated** HuggingFace dataset ``Idavidrein/gpqa`` (accept the license
    on the Hub, then set ``HF_TOKEN`` or run ``huggingface-cli login``).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Benchmarking requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    try:
        ds = load_dataset(
            "Idavidrein/gpqa",
            subset,
            split=split,
            trust_remote_code=True,
            token=_hf_token_for_hub(),
        )
    except Exception as e:
        msg = str(e).lower()
        if "gated" in msg or "authenticated" in msg or "401" in msg:
            raise RuntimeError(
                "GPQA (Idavidrein/gpqa) is gated on the Hugging Face Hub. "
                "Accept the dataset agreement on the model card, then either:\n"
                "  • export HF_TOKEN=<your_read_token>  (or HUGGING_FACE_HUB_TOKEN), or\n"
                "  • huggingface-cli login\n"
                "On Modal, add HF_TOKEN to a Secret and set MODAL_SECRET_NAMES to that secret name."
            ) from e
        raise
    rows = list(ds)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    for i, row in enumerate(rows):
        q = _get(row, "Question", "question")
        correct = _get(row, "Correct Answer", "correct_answer", "CorrectAnswer")
        if q is None or correct is None:
            continue
        wrong = []
        for k in (
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ):
            v = row.get(k)
            if v is not None and str(v).strip():
                wrong.append(str(v))
        if len(wrong) < 3:
            continue
        choices = [str(correct)] + wrong[:3]
        rng = random.Random(seed + i)
        rng.shuffle(choices)
        gold_idx = choices.index(str(correct))
        gold_letter = ("A", "B", "C", "D")[gold_idx]
        ex_id = str(_get(row, "Record ID", "record_id", "id") or i)
        yield ex_id, gold_letter, choices, str(q)


def iter_arc_challenge_mc(
    split: str = "test",
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[tuple[str, str, list[str], str]]:
    """allenai/ai2_arc — ARC-Challenge (4-way MC)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Benchmarking requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=split,
        trust_remote_code=True,
        token=_hf_token_for_hub(),
    )
    rows = list(ds)
    if shuffle:
        random.Random(seed).shuffle(rows)

    for i, row in enumerate(rows):
        q = row.get("question")
        ch = row.get("choices") or {}
        texts = ch.get("text") or []
        labels = ch.get("label") or []
        key = row.get("answerKey")
        if not q or len(texts) < 4 or key is None:
            continue
        letter_map = {str(l).strip().upper(): j for j, l in enumerate(labels)}
        key_u = str(key).strip().upper()
        if key_u in letter_map:
            gold_idx = letter_map[key_u]
        elif str(key).isdigit():
            gold_idx = int(str(key))
            if gold_idx >= 4:
                gold_idx = min(gold_idx - 1, 3)
        elif key in labels:
            gold_idx = labels.index(key)
        else:
            gold_idx = 0
        gold_letter = ("A", "B", "C", "D")[gold_idx]
        choices = [str(t) for t in texts[:4]]
        ex_id = str(row.get("id", i))
        yield ex_id, gold_letter, choices, str(q)


def build_prompt_for_example(
    question: str,
    choices: list[str],
    *,
    system_preamble: str | None = None,
) -> str:
    body = format_mcq_prompt(question, choices[:4])
    if system_preamble:
        return system_preamble.rstrip() + "\n\n" + body
    return body
