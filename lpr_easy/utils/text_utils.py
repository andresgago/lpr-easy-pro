# lpr_easy/utils/text_utils.py
# All comments/docstrings in English.

import re

PAT_BR_OLD = re.compile(r"^[A-Z]{3}\d{4}$")         # ABC1234
PAT_BR_MERC = re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$") # ABC1D23

ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def normalize_plate(raw: str) -> str:
    """
    Normalize OCR output to improve match with Brazilian plate patterns.
    Applies common character disambiguation (O<->0, I/L<->1, S<->5, B<->8, Z<->2).
    Returns uppercase text stripped of separators.
    """
    s = re.sub(r"[^A-Za-z0-9]", "", raw.upper())

    def _try_maps(txt: str) -> str:
        t = (
            txt.replace("O", "0").replace("I", "1").replace("L", "1")
               .replace("S", "5").replace("B", "8").replace("Z", "2")
        )
        if PAT_BR_OLD.match(t) or PAT_BR_MERC.match(t):
            return t
        t2 = (
            txt.replace("0", "O").replace("1", "I")
               .replace("5", "S").replace("8", "B").replace("2", "Z")
        )
        if PAT_BR_OLD.match(t2) or PAT_BR_MERC.match(t2):
            return t2
        return txt

    return _try_maps(s)

def plate_validity_score(s: str) -> int:
    """
    Returns a simple heuristic score for a candidate plate string.
    - Strong bonus for matching known patterns.
    - Bonus for typical length 7.
    - Penalty for characters outside the allowlist (should not happen with allowlist set).
    """
    s = s.strip().upper()
    score = 0
    if PAT_BR_OLD.match(s) or PAT_BR_MERC.match(s):
        score += 10
    if len(s) == 7:
        score += 2
    for ch in s:
        if ch not in ALLOWLIST:
            score -= 1
    return score
