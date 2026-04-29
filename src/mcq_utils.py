from __future__ import annotations

import torch


def letter_choice_token_ids(tokenizer, letters: tuple[str, ...] = ("A", "B", "C", "D")) -> dict[str, int]:
    """
    Map each choice letter to a vocabulary id (typically the last token of a
    short fragment) so we can compare logits after ``Answer:``.
    """
    mapping: dict[str, int] = {}
    for letter in letters:
        chosen: int | None = None
        for frag in (f" {letter}", f"{letter}", f"\n{letter}", f" ({letter})"):
            ids = tokenizer.encode(frag, add_special_tokens=False)
            if not ids:
                continue
            chosen = int(ids[-1])
            if len(ids) == 1:
                break
        if chosen is None:
            raise ValueError(f"Could not tokenize letter {letter!r}")
        mapping[letter] = chosen
    return mapping


def predict_letter_from_logits(
    logits: torch.Tensor,
    letter_to_tid: dict[str, int],
    letters: tuple[str, ...] = ("A", "B", "C", "D"),
) -> str:
    """Pick the letter with highest logit among the candidate token ids (1D ``[V]``)."""
    if logits.dim() != 1:
        raise ValueError(f"expected 1D logits [vocab], got shape {tuple(logits.shape)}")
    best_letter = letters[0]
    best_score = logits[letter_to_tid[best_letter]]
    for letter in letters[1:]:
        s = logits[letter_to_tid[letter]]
        if s > best_score:
            best_score = s
            best_letter = letter
    return best_letter


def predict_letters_from_logits(
    logits: torch.Tensor,
    letter_to_tid: dict[str, int],
    letters: tuple[str, ...] = ("A", "B", "C", "D"),
) -> list[str]:
    """Batched ``[B, vocab]`` logits → one letter prediction per row."""
    if logits.dim() == 1:
        return [predict_letter_from_logits(logits, letter_to_tid, letters)]
    if logits.dim() != 2:
        raise ValueError(f"expected logits [vocab] or [B, vocab], got {tuple(logits.shape)}")
    return [
        predict_letter_from_logits(logits[i], letter_to_tid, letters)
        for i in range(logits.shape[0])
    ]


def format_mcq_prompt(
    question: str,
    choices: list[str],
    letters: tuple[str, ...] = ("A", "B", "C", "D"),
) -> str:
    lines = [question.strip(), "", "Options:"]
    for letter, text in zip(letters, choices):
        lines.append(f"{letter}. {text.strip()}")
    lines.extend(
        [
            "",
            "Respond with only the letter of the correct option (A, B, C, or D).",
            "Answer:",
        ]
    )
    return "\n".join(lines)
