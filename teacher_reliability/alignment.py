"""Conservative character-offset alignment for generated token IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenAlignment:
    text: str
    character_spans: list[tuple[int, int] | None] | None
    status: str


def align_generated_tokens(token_ids: list[int], tokenizer: Any) -> TokenAlignment:
    """Map visible generated IDs to text offsets only after an exact round trip."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    visible_positions = [
        index for index, token_id in enumerate(token_ids) if token_id not in special_ids
    ]
    visible_ids = [token_ids[index] for index in visible_positions]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        round_trip_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if round_trip_ids and isinstance(round_trip_ids[0], list):
            round_trip_ids = round_trip_ids[0]
            offsets = offsets[0]
    except (KeyError, TypeError, ValueError, NotImplementedError):
        return TokenAlignment(text, None, "offsets_unavailable")

    if list(round_trip_ids) != visible_ids or len(offsets) != len(visible_ids):
        return TokenAlignment(text, None, "token_roundtrip_mismatch")

    character_spans: list[tuple[int, int] | None] = [None] * len(token_ids)
    for position, offset in zip(visible_positions, offsets):
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start or end > len(text):
            return TokenAlignment(text, None, "invalid_offsets")
        character_spans[position] = (start, end)
    return TokenAlignment(text, character_spans, "aligned")
