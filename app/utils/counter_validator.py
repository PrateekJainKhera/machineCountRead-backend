from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ValidationResult:
    """Result of validating a new OCR counter reading."""
    is_valid: bool
    confidence_ok: bool        # confidence passed the threshold
    rate_ok: bool              # value change is physically possible
    direction_ok: bool         # counter didn't go backwards unexpectedly
    digit_count_ok: bool       # digit count matches previous reading
    reason: str                # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "confidence_ok": self.confidence_ok,
            "rate_ok": self.rate_ok,
            "direction_ok": self.direction_ok,
            "digit_count_ok": self.digit_count_ok,
            "reason": self.reason,
        }


def validate_reading(
    new_value: int,
    new_confidence: float,
    prev_value: Optional[int],
    prev_timestamp: Optional[datetime],
    current_timestamp: datetime,
    min_confidence: float = 0.85,
    max_rate_per_second: float = 5.0,
) -> ValidationResult:
    """
    Validate a new OCR counter reading against confidence and physical rate limits.

    Two checks are applied:

    1. Confidence gate
       The OCR confidence must be >= min_confidence.
       Below this, the reading is too uncertain to trust — mid-transition digits,
       glare, or blur can produce plausible-looking but wrong numbers.

    2. Rate validation
       The counter change from the previous accepted reading must be physically
       possible given the time elapsed and the machine's max speed.

       Formula:
           max_allowed_change = max_rate_per_second × elapsed_seconds × 1.2  (20% buffer)
           If new_value - prev_value > max_allowed_change → reject

       Also catches counter going backwards:
           If new_value < prev_value → rejected as impossible
           (counters only reset to 0 — a drop by any other amount = bad OCR read)

    Args:
        new_value:            The OCR-read counter value to validate.
        new_confidence:       OCR confidence score (0.0–1.0).
        prev_value:           Last accepted counter value (None if no history yet).
        prev_timestamp:       When prev_value was accepted (None if no history).
        current_timestamp:    Timestamp of the new reading.
        min_confidence:       Minimum acceptable confidence (default 0.85).
        max_rate_per_second:  Machine max sheets per second (default 5.0).
                              RMGT 11,000 SPH ÷ 3600 = 3.05/sec → use 5.0 with buffer.

    Returns:
        ValidationResult with is_valid=True if both checks pass.
    """

    # ── Check 1: Confidence gate ──────────────────────────────────────
    confidence_ok = new_confidence >= min_confidence

    if not confidence_ok:
        return ValidationResult(
            is_valid=False,
            confidence_ok=False,
            rate_ok=True,
            direction_ok=True,
            digit_count_ok=True,
            reason=(
                f"Confidence {new_confidence:.3f} is below threshold {min_confidence}. "
                f"Reading rejected — digit may be mid-transition or obscured."
            ),
        )

    # ── No previous reading — nothing to compare against ─────────────
    if prev_value is None or prev_timestamp is None:
        return ValidationResult(
            is_valid=True,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=True,
            digit_count_ok=True,
            reason="First reading — no rate comparison available. Accepted.",
        )

    elapsed_seconds = (current_timestamp - prev_timestamp).total_seconds()
    delta = new_value - prev_value

    # ── Check 2a: Direction — counter must not go backwards ───────────
    # ONLY small backwards deltas (OCR noise on the last digit) are accepted.
    # EVERY larger backwards jump is rejected here — including drops to near
    # zero. A partial read of a soft frame ("1359" → "39") looks exactly like
    # a counter reset, so resets get NO free pass on a single reading.
    # Real resets (job change → counter back to 0, counting up) are adopted by
    # the service-level rejection-streak recovery after several consecutive
    # consistent readings; partial reads don't repeat, so they never qualify.
    noise_tolerance = 20  # OCR can misread last 1-2 digits (e.g. 18929→18917 is noise/loop)
    direction_ok = delta >= -noise_tolerance

    if not direction_ok:
        return ValidationResult(
            is_valid=False,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=False,
            digit_count_ok=True,
            reason=(
                f"Counter went backwards: {prev_value} → {new_value} (delta={delta}). "
                f"Rejected — partial read or reset; consistent repeats will be adopted."
            ),
        )

    # ── Check 2b: Noise recovery — prev baseline poisoned by bad OCR ─
    # Must run BEFORE digit count check so a bad low-digit baseline (e.g. prev=2)
    # doesn't permanently block all correct high-digit reads (e.g. new=18917).
    # Happens when: display shows "19,031", comma splits the detection into "19"
    # and "031", and the small fragment gets accepted first. Subsequent correct
    # reads (19031) then look like impossible jumps from 105/190.
    # Rule: if prev < 500 AND new > 1000 AND new confidence >= 0.90, trust the
    # high-confidence read and reset the baseline.
    coming_from_reset = prev_value < 100
    prev_was_likely_noise = (
        prev_value < 500
        and new_value > 1000
        and new_confidence >= 0.90
    )
    if prev_was_likely_noise:
        return ValidationResult(
            is_valid=True,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=True,
            digit_count_ok=True,
            reason=(
                f"Previous value {prev_value} likely OCR noise (fragment of larger number). "
                f"New reading {new_value} at conf={new_confidence:.2f} accepted as new baseline."
            ),
        )
    # ── Check 2c: Digit count — must match previous reading ──────────
    # A 5-digit counter reading as 3 digits is always an OCR error.
    # Example: prev=18944 (5 digits), new=111 (3 digits) → reject.
    # Allow ±1 digit for natural rollover: 99999 (5) → 100001 (6) is valid.
    # Skip this check on reset (counter returning to near-zero is a job reset).
    prev_digit_count = len(str(prev_value))
    new_digit_count  = len(str(new_value))
    digit_count_ok   = abs(new_digit_count - prev_digit_count) <= 1

    if not digit_count_ok:
        return ValidationResult(
            is_valid=False,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=True,
            digit_count_ok=False,
            reason=(
                f"Digit count mismatch: prev={prev_value} ({prev_digit_count} digits) "
                f"→ new={new_value} ({new_digit_count} digits). "
                f"Rejected — OCR likely read a partial or wrong number."
            ),
        )

    # ── Check 2d: Rate — change must be physically possible ──────────
    max_allowed = max_rate_per_second * elapsed_seconds * 1.2 if elapsed_seconds > 0 else 0
    # Only rate-check deltas above the noise band — small deltas are OCR noise
    # on the same counter (e.g. 18917 → 18929 in 0.5s). Band is 25: wide enough
    # for digit noise, narrow enough that a +40 digit-flip misread (1741→1781)
    # cannot slip under it when the machine's max rate says it's impossible.
    ocr_noise_band = 25
    if elapsed_seconds > 0 and not coming_from_reset and delta > ocr_noise_band:
        actual_rate = delta / elapsed_seconds
        rate_ok = delta <= max_allowed

        if not rate_ok:
            return ValidationResult(
                is_valid=False,
                confidence_ok=True,
                rate_ok=False,
                direction_ok=True,
                digit_count_ok=True,
                reason=(
                    f"Counter jumped {delta} in {elapsed_seconds:.1f}s "
                    f"({actual_rate:.1f}/sec) — exceeds machine max "
                    f"({max_rate_per_second}/sec × 1.2 buffer = {max_allowed:.0f} max). "
                    f"Rejected — likely OCR misread."
                ),
            )
    else:
        rate_ok = True

    # ── All checks passed ─────────────────────────────────────────────
    actual_rate = delta / elapsed_seconds if elapsed_seconds > 0 else 0
    return ValidationResult(
        is_valid=True,
        confidence_ok=True,
        rate_ok=True,
        direction_ok=True,
        digit_count_ok=True,
        reason=(
            f"Valid. Counter: {prev_value} → {new_value} "
            f"(+{delta} in {elapsed_seconds:.1f}s, {actual_rate:.1f}/sec)."
        ),
    )
