"""Stateless, X-only OCR prompt representation; see assets/feature_schema.json.

No corpus fitting, target access, randomness, downloads, or candidate identifiers.
The only read is the bundled, fixed feature schema.
"""
import json
import math
from pathlib import Path
import re
import string

import numpy as np
import torch


_SCHEMA = json.loads((Path(__file__).parent / "assets/feature_schema.json").read_text())
DIMENSION = _SCHEMA["dimension"]
_CHARACTERS = {c: i for i, c in enumerate(string.ascii_lowercase + string.digits)}
_PUNCTUATION = frozenset(string.punctuation)
_TOKENS = re.compile(r"[a-z0-9]+")
_WORDS = re.compile(r"\S+")
_PHRASES = tuple(
    tuple(tuple(_TOKENS.findall(phrase.lower())) for phrase in group["phrases"])
    for group in _SCHEMA["context"]
)
_PHRASE_LENGTHS = tuple(sorted({len(p) for group in _PHRASES for p in group}))
_BURDEN_SCALE = math.sqrt(DIMENSION * _SCHEMA["weights"]["burden"] / 16)
_SPECTRUM_SCALE = math.sqrt(DIMENSION * _SCHEMA["weights"]["spectrum"] / 2)
_CONTEXT_SCALE = math.sqrt(DIMENSION * _SCHEMA["weights"]["context"])


def _saturate(count, reference):
    return count / (count + reference)


def _normalize_in_place(block, scale):
    norm = math.sqrt(float(np.dot(block, block)))
    if norm > 0:
        block *= scale / norm


def _encode_prompt(prompt, output):
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    parts = prompt.split('"')
    raw_target = parts[1] if len(parts) > 1 else ""
    target = raw_target.replace(" ", "").lower()
    # Target wording must not be mistaken for a rendering instruction.
    context = " ".join(parts[::2]).lower()
    tokens = _TOKENS.findall(context)
    words = _WORDS.findall(raw_target)
    n = len(target)
    quotes = len(parts) - 1
    extra = max(0, quotes // 2 - 1)
    output[:16] = (
        _saturate(n, 16),
        _saturate(len(words), 4),
        _saturate(max((len(w) for w in words), default=0), 10),
        sum(c in string.digits for c in target) / max(1, n),
        sum(c in _PUNCTUATION for c in target) / max(1, n),
        sum(ord(c) > 127 for c in target) / max(1, n),
        sum(c.isupper() for c in raw_target) / max(1, sum(c.isalpha() for c in raw_target)),
        raw_target.count(" ") / max(1, len(raw_target)),
        len(set(target)) / max(1, n),
        sum(a == b for a, b in zip(target, target[1:])) / max(1, n - 1),
        _saturate(len(prompt), 240),
        _saturate(len(tokens), 40),
        _saturate(extra, 2),
        float(quotes == 0),
        float(quotes > 0 and n == 0),
        float(quotes % 2 != 0),
    )
    output[:16] *= _BURDEN_SCALE
    classes = [_CHARACTERS.get(c, 36 if c in _PUNCTUATION else 37) for c in target]
    for symbol in classes:
        output[16 + symbol] += 1.0
    for left, right in zip(classes, classes[1:]):
        output[54 + 38 * left + right] += 1.0
    _normalize_in_place(output[16:54], _SPECTRUM_SCALE)
    _normalize_in_place(output[54:1498], _SPECTRUM_SCALE)

    # Sets are used only for membership, never to determine coordinate order.
    sequences = {
        length: {tuple(tokens[i:i + length]) for i in range(len(tokens) - length + 1)}
        for length in _PHRASE_LENGTHS
    }
    for i, phrases in enumerate(_PHRASES):
        output[1498 + i] = float(any(phrase in sequences[len(phrase)] for phrase in phrases))
    _normalize_in_place(output[1498:1530], _CONTEXT_SCALE)


def encode_rows(rows, *, dtype=None, device=None):
    """Return finite float64 (len(rows), 1530), preserving order and duplicates.

    Missing, empty and unmatched quote cases have explicit finite indicators;
    this is representation robustness, not validation of the OCR scorer input.
    """
    if dtype is not None and dtype != torch.double:
        raise ValueError("The representation contract requires torch.double")
    values = np.zeros((len(rows), DIMENSION), dtype=np.float64)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "prompt" not in row:
            raise TypeError("Each row must be a dictionary containing prompt")
        _encode_prompt(row["prompt"], values[index])
    return torch.as_tensor(values, dtype=torch.double, device=device)
