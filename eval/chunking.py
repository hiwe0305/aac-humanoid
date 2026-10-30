"""Which part of the predicted action chunk gets executed before re-planning.

Baseline (this stage): fixed horizon h  (GR00T default h=16; paper also reports h = 2, 4, 8, 12).
`select_horizon` is the single hook where an adaptive rule (AAC) can be plugged in later.
"""


def select_horizon(chunk: dict, fixed_h: int) -> int:
    chunk_len = len(next(v for k, v in chunk.items() if k.startswith("action.")))
    return max(1, min(fixed_h, chunk_len))

