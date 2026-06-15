"""Job card value helpers — digit-core identity + clean-form scoring."""

import re

# Typical job number: 1-3 uppercase letters, optional dash, then digits
# (e.g. JC-4521, PB-1088, MX-77). A misread like "JBCA4521" has too many
# leading letters and scores lower, so the clean form wins on merge.
_GOOD = re.compile(r"^[A-Z]{1,3}-?\d{2,}$")
_LEADING_LETTERS = re.compile(r"^([A-Z]*)")


def core(v) -> str:
    """Digit-only identity — stable across reads where prefix letters flicker."""
    return "".join(ch for ch in (v or "") if ch.isdigit())


def score(v) -> int:
    """Higher = more like a clean job number. Used to pick the best textual form."""
    if not v:
        return -1
    s = 0
    if _GOOD.match(v):
        s += 100
    if "-" in v:
        s += 5
    # Penalize long letter runs ("JBCA" = 4 → bad; "JC" = 2 → good)
    s -= len(_LEADING_LETTERS.match(v).group(1))
    return s
