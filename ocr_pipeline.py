"""
Renewal Shield — Phase 3, step 1: OCR + clause splitting + field extraction.

Reads a contract image, turns it into text, splits the "Additional Terms"
section into separate clauses and pulls out the key fields (rent, deposit,
dates, cheques...). Then compares everything against the generator's labels.

Two OCR engines:
  tesseract  – classic OCR (needs the Tesseract program installed)
  rapidocr   – deep-learning OCR, pip-only (pip install rapidocr-onnxruntime)

Usage:
  python ocr_pipeline.py --data out --engine rapidocr
  python ocr_pipeline.py --data out --engine tesseract --limit 10
  python ocr_pipeline.py --image out/photo/contract_0001.png --engine rapidocr   # one file
  python ocr_pipeline.py --data out --rescore      # re-score saved OCR output, no OCR (seconds)
"""

import argparse
import csv
import difflib
import glob
import json
import os
import re
import time

from PIL import Image

# ----------------------------------------------------------------- OCR engines
_rapid = None
RAPID_PARAMS = {"text_score": 0.0, "box_thresh": 0.5, "unclip_ratio": 1.6}


def ocr_tesseract(img):
    import pytesseract
    # On Windows, if tesseract isn't on PATH, uncomment and fix the path:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    img = deskew(img.convert("RGB"))          # same straightening as RapidOCR, for a fair comparison
    if img.width < 1500:                       # phone uploads are small: upscale for Tesseract
        f = 1600 / img.width
        img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
    return pytesseract.image_to_string(img, config="--psm 4")


def _rapid_engine():
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid = RapidOCR()
    return _rapid


def estimate_tilt(img):
    """Page tilt in degrees, from the angle of detected text lines (detection only, fast)."""
    import math
    import numpy as np
    boxes, _ = _rapid_engine()(np.array(img.convert("RGB")), use_rec=False, use_cls=False)
    if not boxes:
        return 0.0
    angles = []
    for box in boxes:
        box = box[0] if len(box) == 3 else box
        (x0, y0), (x1, y1) = box[0], box[1]
        if x1 - x0 > 150:
            angles.append(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    return sorted(angles)[len(angles) // 2] if angles else 0.0


def deskew(img):
    angle = estimate_tilt(img)
    if abs(angle) < 0.2:
        return img
    bg = tuple(int(v) for v in img.convert("RGB").resize((1, 1)).getpixel((0, 0)))
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg)


def _line_score(img, boxes, eng):
    """Mean recognition confidence of a few long text lines (no angle classifier),
    used to tell an upright page from an upside-down one."""
    import numpy as np
    scores = []
    for x0, y0, x1, y1 in boxes:
        rr, _ = eng(np.array(img.crop((x0, y0, x1, y1))), use_det=False, use_cls=False)
        got = _rec_result(rr)
        scores.append(got[1] if got else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _aligned_share(edges, width):
    """Share of line edges that sit within 1.5% of page width of the most common edge position."""
    tol = 0.015 * width
    return max(sum(abs(e - c) <= tol for e in edges) for c in edges) / len(edges)


def _upside_down(img, boxes, eng):
    """Is this (horizontal-lined) page upside down?
    Layout first: contract text is left-aligned, so many lines share a left edge and right edges are
    ragged. Upside down, that flips. The recognizer is only the tie-breaker (justified text), because
    it often scores upside-down crops the same as upright ones."""
    if len(boxes) >= 6:
        W = img.size[0]
        left = _aligned_share([b[0] for b in boxes], W)
        right = _aligned_share([b[2] for b in boxes], W)
        if abs(left - right) >= 0.1:
            return right > left
    longest = sorted(boxes, key=lambda b: b[0] - b[2])[:8]
    if not longest:
        return False
    W, H = img.size
    flipped = [(W - x1, H - y1, W - x0, H - y0) for x0, y0, x1, y1 in longest]
    return _line_score(img.rotate(180), flipped, eng) > _line_score(img, longest, eng) + 0.1


def fix_orientation(img):
    """Turn a photo so the text lines are horizontal and the right way up.

    Real phone photos are often sideways (taken in landscape, or EXIF rotation lost when
    sent over WhatsApp) or strongly tilted. deskew() only handles small tilts, so:
      1. detect text boxes on a small copy; the direction of their long side = line direction,
         at any angle (sideways = about 90 degrees);
      2. rotate so lines are horizontal;
      3. that leaves 'upright or upside down': read a few long lines both ways, keep the better.
    Upright pages (all synthetic data) come out unchanged."""
    import math
    import numpy as np
    eng = _rapid_engine()
    img = img.convert("RGB")
    scale = min(1.0, 1000 / max(img.size))
    small = img.resize((int(img.width * scale), int(img.height * scale))) if scale < 1 else img
    boxes, _ = eng(np.array(small), use_rec=False, use_cls=False)
    if not boxes:
        return img
    angles = []                                   # (angle of the long side, length)
    for box in boxes:
        box = box[0] if len(box) == 3 else box
        p = [tuple(map(float, q)) for q in box]
        e1 = (p[1][0] - p[0][0], p[1][1] - p[0][1])
        e2 = (p[2][0] - p[1][0], p[2][1] - p[1][1])
        long_e, short_e = (e1, e2) if math.hypot(*e1) >= math.hypot(*e2) else (e2, e1)
        L, S = math.hypot(*long_e), math.hypot(*short_e)
        if L > 40 and L > 2.5 * S:                # clearly a text line, not a blob
            a = math.degrees(math.atan2(long_e[1], long_e[0]))
            a = (a + 90) % 180 - 90               # direction only: map into [-90, 90)
            angles.append((a, L))
    if not angles:
        return img
    # length-weighted median angle
    angles.sort()
    half, acc, theta = sum(l for _, l in angles) / 2, 0.0, 0.0
    for a, l in angles:
        acc += l
        if acc >= half:
            theta = a
            break

    def rotate(im, deg):
        if abs(deg) < 0.2:
            return im
        bg = tuple(int(v) for v in im.resize((1, 1)).getpixel((0, 0)))
        return im.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=bg)

    cand = rotate(small, theta)
    # long, horizontal lines of the rotated small copy -> compare upright vs upside down
    b2, _ = eng(np.array(cand), use_rec=False, use_cls=False)
    lines = []
    for box in b2 or []:
        box = box[0] if len(box) == 3 else box
        xs, ys = [q[0] for q in box], [q[1] for q in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 - x0 > 3 * (y1 - y0) and x1 - x0 > 60:
            lines.append((x1 - x0, (int(x0), int(y0), int(x1), int(y1))))
    if _upside_down(cand, [b for _, b in lines], eng):
        theta += 180
    # Small tilts are left to deskew(): rotating twice blurs the digits ("47,000" -> "47,ooo").
    if abs(theta) < 10:
        return img
    return rotate(img, theta)


def _rec_result(rr):
    """Recognition-only output differs between RapidOCR versions:
    [[box, text, score]] on some, [[text, score]] on others. Return (text, score) or None."""
    if not rr or not rr[0]:
        return None
    item = rr[0]
    try:
        if len(item) >= 3:
            return str(item[1]), float(item[2])
        if len(item) == 2:
            return str(item[0]), float(item[1])
    except (TypeError, ValueError):
        pass
    return None


def _rec_in_pieces(crop, eng, parts):
    """Read a long line in pieces, cut at the whitest column (a gap between words) near each split point.
    Some RapidOCR versions (1.4.x) return nothing for very wide line crops that 1.2.x reads fine."""
    import numpy as np
    g = np.asarray(crop.convert("L"), dtype=float)
    ink = (255 - g).sum(axis=0)                     # dark pixels per column
    W = g.shape[1]
    cuts = [0]
    for i in range(1, parts):
        lo, hi = int(W * (i / parts - 0.1)), int(W * (i / parts + 0.1))
        cuts.append(lo + int(np.argmin(ink[lo:hi])))
    cuts.append(W)
    texts, scores = [], []
    for a, b in zip(cuts, cuts[1:]):
        rr, _ = eng(np.asarray(crop.crop((a, 0, b, crop.height))), use_det=False, use_cls=False)
        got = _rec_result(rr)
        if not got or not got[0].strip():
            return None
        texts.append(got[0].strip())
        scores.append(got[1])
    return " ".join(texts), sum(scores) / len(scores)


def ocr_rapid(img):
    """Deskew the page, run RapidOCR, retry weak lines, then rebuild lines in reading order."""
    import numpy as np
    eng = _rapid_engine()
    img = deskew(fix_orientation(img.convert("RGB")))
    arr = np.array(img)
    result, _ = eng(arr, **RAPID_PARAMS)
    if not result:
        return ""
    items, heights = [], []
    retries_left = 15
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        # retry low-confidence lines from a padded crop (fixes long lines read as garbage).
        # Budget per page: on a hopelessly blurry photo every line is low-confidence and retrying
        # them all took 2 minutes for nothing.
        if float(score) < 0.75 and x1 - x0 > 150 and retries_left > 0:
            retries_left -= 1
            for px, py in [(10, 6), (20, 12), (30, 18)]:
                crop = img.crop((max(0, x0 - px), max(0, y0 - py), x1 + px, y1 + py))
                rr, _ = eng(np.array(crop), use_det=False, use_cls=False)
                got = _rec_result(rr)
                if got and got[1] > float(score):
                    text, score = got
                if float(score) >= 0.85:
                    break
            # still bad AND far too few characters for the line's width ("ay" for a whole line):
            # the recognizer gave up on a wide crop, so read it in 2, then 3 pieces
            if float(score) < 0.75 and len(text.strip()) < (x1 - x0) / (2 * max(1, y1 - y0)):
                crop = img.crop((max(0, x0 - 10), max(0, y0 - 6), x1 + 10, y1 + 6))
                for parts in (2, 3):
                    got = _rec_in_pieces(crop, eng, parts)
                    if got and got[1] > float(score):
                        text, score = got
                    if float(score) >= 0.85:
                        break
        if float(score) < 0.3 or not text.strip():
            continue
        heights.append(y1 - y0)
        items.append(((y0 + y1) / 2, x0, text.strip()))
    tol = 0.5 * (sorted(heights)[len(heights) // 2] if heights else 20)
    items.sort()
    lines, cur, cur_y = [], [], None
    for y, x, t in items:
        if cur_y is None or abs(y - cur_y) <= tol:
            cur.append((x, t))
            cur_y = y if cur_y is None else (cur_y * (len(cur) - 1) + y) / len(cur)
        else:
            lines.append(" ".join(t for _, t in sorted(cur)))
            cur, cur_y = [(x, t)], y
    if cur:
        lines.append(" ".join(t for _, t in sorted(cur)))
    return "\n".join(lines)


ENGINES = {"tesseract": ocr_tesseract, "rapidocr": ocr_rapid}


# ----------------------------------------------------------------- preprocessing
def remove_colored_ink(img, red_thresh=10, blue_thresh=20, dark=120):
    """Erase coloured marks (red watermark, blue signatures, stamps) before OCR.
    Printed text is dark, so any pixel darker than `dark` is always kept, even if
    it has a red tint from the watermark on top of it. Light pixels that are
    reddish (R > G) or bluish (B > R) are painted with the page's paper colour."""
    import numpy as np
    a = np.array(img.convert("RGB")).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    light = a.mean(axis=2) > dark
    reddish = (r - g) > red_thresh
    bluish = ((b - r) > blue_thresh) & ((b - g) > 5)
    mask = light & (reddish | bluish)
    paper = np.median(a[light & ~mask], axis=0) if (light & ~mask).any() else np.array([255, 255, 255])
    out = a.copy()
    out[mask] = paper
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ----------------------------------------------------------------- spacing repair
# RapidOCR's default model sometimes drops spaces in English ("berefunded").
# We re-split glued words with a small vocabulary: common English + every word in
# the clause bank. A token is only split if ALL its pieces are known words.
COMMON_WORDS = """a about above after again against all also an and any are as at be been before
being below between both but by can could did do does down during each either even every few for
from further had has have having he her here him his how i if in into is it its itself just less
may more most must my no nor not now of off on once one only or other our out over own per same
shall she should so some such than that the their them then there these they this those through
to too two under until up upon very was we were what when where whether which while who whom why
will with within without would you your three four five six seven eight nine ten eleven twelve
first second third day days month months year years time times date dates notice notices written
writing rent rents tenant tenants landlord landlords contract contracts lease property premises
deposit amount amounts cost costs pay paid payment payments cheque cheques bank transfer fee fees
law laws repair repairs maintenance renewal renew expiry terms term clause clauses party parties
use used new end start early late full part whole period penalty charge charges vacate vacating
increase increases decrease market rate rates index rental refund refunded refundable return
returned agreed agree accept accepted damage damages wear tear normal good condition responsible
""".split()

_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        words = set(COMMON_WORDS)
        for path in sorted(glob.glob("renewal_shield_clause_bank_*.xlsx")):
            try:
                import pandas as pd
                df = pd.read_excel(path, sheet_name="Clause Bank").fillna("")
                for col in ("clause_text", "edited_text"):
                    for t in df[col].astype(str):
                        words.update(re.findall(r"[a-z]+", t.lower()))
            except Exception:
                pass
        _VOCAB = words
    return _VOCAB


def _segment(token, vocab):
    """Split a glued lowercase word into known words (fewest pieces). None if impossible."""
    n = len(token)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            if best[j] is None:
                continue
            piece = token[j:i]
            if piece in vocab and (len(piece) > 1 or piece in ("a", "i")):
                cand = best[j] + [piece]
                if best[i] is None or len(cand) < len(best[i]):
                    best[i] = cand
    return best[n]


# Cyrillic/Greek letters that look identical to Latin ones (the OCR model can output them)
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "ο": "o", "α": "a", "ν": "v", "ι": "i", "Ο": "O", "Α": "A", "Β": "B",
    "Ε": "E", "Τ": "T", "Ν": "N", "Μ": "M", "Κ": "K", "Ι": "I",
    "‐": "-", "‑": "-", "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "", "\u00a0": " ",
})


def normalize_chars(text):
    import unicodedata
    return unicodedata.normalize("NFKC", text).translate(HOMOGLYPHS)


def _split_word(word, vocab):
    """Split one glued alphabetic word if every piece is a known word; keep capitalisation."""
    if len(word) < 5 or word.lower() in vocab:
        return word
    parts = _segment(word.lower(), vocab)
    if not parts or len(parts) < 2:
        return word
    pieces, k = [], 0
    for p in parts:
        pieces.append(word[k:k + len(p)])
        k += len(p)
    return " ".join(pieces)


def fix_spacing(text):
    """Repair missing spaces in OCR output."""
    text = normalize_chars(text)
    t = re.sub(r"([a-z])\(", r"\1 (", text)          # thirty(30) -> thirty (30)
    t = re.sub(r"\)([A-Za-z])", r") \1", t)          # (30)days -> (30) days
    t = re.sub(r"([,;])([A-Za-z])", r"\1 \2", t)      # vacating,less -> vacating, less
    t = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", t)    # law.The -> law. The
    t = re.sub(r"(\d)\.([A-Z])", r"\1. \2", t)      # 2.The -> 2. The
    t = re.sub(r"s'([a-z])", r"s' \1", t)             # months'written -> months' written
    vocab = _vocab()
    out = []
    for tok in t.split():
        m = re.match(r"^([^A-Za-z]*)([A-Za-z][A-Za-z-]*[A-Za-z]|[A-Za-z])([^A-Za-z]*)$", tok)
        if not m:
            out.append(tok)
            continue
        core = m.group(2)
        if "-" in core:
            # only the first part can be glued ("arere-let" -> "are re-let")
            head, rest = core.split("-", 1)
            core = _split_word(head, vocab) + "-" + rest
        else:
            core = _split_word(core, vocab)
        out.append(m.group(1) + core + m.group(3))
    return " ".join(out)


def strip_trailing_scrap(t):
    """Drop a short leftover after the clause's final full stop (e.g. 'Sr' from the watermark)."""
    m = re.match(r"^(.*[.)])\s+(\S{1,6})$", t)
    return m.group(1) if m and not m.group(2).endswith(".") else t


# ----------------------------------------------------------------- clause split
START = re.compile(r"ADD[I1l]T[I1l]ONAL\W*TERMS?", re.I)          # "ADDITIONAL TERMS", "ADDITIONAL.TERMS", ...
START_FALLBACK = re.compile(r"^.*\bTERMS\b(?!\s*AND).*$|^.*ADD[I1l]T\w*.*$", re.M)   # badly read heading
CHEQUES_LINE = re.compile(r"^.*Cheques.*$", re.M | re.I)        # last field line before the clauses
STOP = re.compile(r"(Landlord\s*Signature|Tenant\s*Signature|SYNTHETIC\s*SAMPLE\s*[—-])", re.I)
SIG_SCRAP = re.compile(r"\s*\b(Land\w*|Ten\w*)\s*S[il1][a-z]*\s*$")   # "LandlordSiamna", "Tenaat Sign"
# clause numbers as OCR really reads them: "2. The", "2.The", "1..The", "2: The", "3, The"
ITEM = re.compile(r"(?:^|\n)\s*(\d{1,2})\s*[.):,;]{1,2}\s*(?=[A-Z])")


WM = "SYNTHETICSAMPLE"


def strip_watermark(t):
    """Drop all-caps fragments of the 'SYNTHETIC SAMPLE' watermark (e.g. 'SYNTH', 'AMPLE')."""
    words = [w for w in t.split()
             if not (len(w) >= 3 and w.strip(".,;:").isupper() and w.strip(".,;:") in WM)]
    return " ".join(words).strip()


# line begins with a clause number as OCR reads it: "3.", "3:", "3,", "3..", or with a stray mark "4.7 The"
ITEM_START = re.compile(r"^\s*(\d{1,2})\s*[.):,;]{1,2}\s*(?:[^\sA-Za-z]{1,2}\s+)?(?=[A-Z]|l[a-z])")   # "2.lf" = "2. If"


def _clause_lines(section):
    """Group the section's lines into clauses.

    A new clause starts when a line
      (a) begins with a clause number ("3.", "3:", "3,"), or
      (b) begins with a capital letter right after a SHORT line.
    (b) catches clauses whose number the OCR dropped: a clause's last line is usually
    shorter than a full line, while a wrapped continuation follows a full-width line.
    """
    lines = [ln.strip() for ln in section.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 4]                 # drop OCR scraps like "M"
    if not lines:
        return []
    full = max(len(ln) for ln in lines)
    groups, prev = [], None
    for ln in lines:
        numbered = ITEM_START.match(ln)
        after_short = (prev is not None and len(prev) < 0.75 * full and ln[:1].isupper())
        if numbered or after_short or not groups:
            groups.append([ITEM_START.sub("", ln) if numbered else ln])
        else:
            groups[-1].append(ln)
        prev = ln
    return [" ".join(g) for g in groups]


def _is_signature_line(line):
    """Signature block, even when OCR mangles it: 'Landlurd Sigaatury', 'Laodlord Signabure', 'Landlord Sia launnts'.
    Real contracts end with a signature block too, so this is not specific to the synthetic layout."""
    words = re.findall(r"[A-Za-z]{3,}", line)
    sim = lambda w, target: difflib.SequenceMatcher(None, w.lower(), target).ratio()
    if len(line) > 60:                           # clause lines are long; signature lines are short
        return False
    if {w.lower() for w in words} & {"the", "of", "this", "shall", "by", "to", "and", "upon", "any", "is", "be"}:
        return False                             # reads like a sentence, not a signature block
    if any(len(w) >= 6 and sim(w, "signature") >= 0.7 for w in words):
        return True
    # short line: mangled "Landlord"/"Tenant" + a word starting like "Si..." ("Landlord Sia launnts"),
    # but not ordinary clause words ("Tenant shall", "Landlord's standard")
    return (len(line) < 40 and len(words) >= 2 and words[1][:2].lower() in ("si", "sl", "s1", "sg")
            and (sim(words[0], "landlord") >= 0.7 or sim(words[0], "tenant") >= 0.7))


def split_clauses(text):
    """Return the list of clause texts from the Additional Terms section."""
    m = START.search(text) or START_FALLBACK.search(text)
    if not m:                                   # heading unreadable: start after the last field line
        cheq = list(CHEQUES_LINE.finditer(text))
        m = cheq[-1] if cheq else None
    section = text[m.end():] if m else text
    s = STOP.search(section)
    if s:
        section = section[:s.start()]
    kept = []
    for ln in section.splitlines():              # stop at a (mangled) signature line
        if _is_signature_line(ln):
            break
        kept.append(ln)
    section = "\n".join(kept)
    clauses = []
    for t in _clause_lines(section):
        t = " ".join(t.split())
        t = strip_trailing_scrap(SIG_SCRAP.sub("", fix_spacing(strip_watermark(t))))
        t = re.sub(r"^l(?=[a-z])", "I", t)            # "lf the Tenant" -> "If the Tenant"
        if len(t) > 15:
            clauses.append(t)
    return clauses


# ----------------------------------------------------------------- fields
def _num(s):
    s = re.sub(r"[^\d]", "", (s or "").translate(str.maketrans("oOD", "000")))   # OCR reads 0 as o/O/D
    return int(s) if s else None


def _date(s):
    """'15/07/2027' -> '2027-07-15'. Also when OCR drops a slash ('15/072027', '1105/2026')."""
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", s or "")
    if m:
        d, mo, y = m.groups()
    else:
        digits = re.sub(r"\D", "", s or "")
        if len(digits) != 8:
            return None
        d, mo, y = digits[:2], digits[2:4], digits[4:]
    if not (1 <= int(d) <= 31 and 1 <= int(mo) <= 12 and 1990 <= int(y) <= 2100):
        return None
    return f"{y}-{int(mo):02d}-{int(d):02d}"


FIELD_PATTERNS = {
    "contract_date": (r"Con\w{2,6}\s*Date\W*([\d/ ]{8,12})", _date),
    "start_date": (r"Start\s*Date\W*([\d/ ]{8,12})", _date),
    "end_date": (r"End\s*Date\W*([\d/ ]{8,12})", _date),
    "annual_rent": (r"Annual\s*Rent\w*\W*A[A-Z]D\s*(\d[\doO,. ]*)", _num),
    "contract_value": (r"Contract\s*Valu\w*\W*A[A-Z]D\s*(\d[\doO,. ]*)", _num),
    "security_deposit": (r"Security\s*Dep\w*\W*A[A-Z]D\s*(\d[\doO,. ]*)", _num),
    "num_cheques": (r"No\W*of\s*Cheques\W*(\d+|N/?A)",
                    lambda s: 0 if "N" in s.upper() else _num(s)),
    "payment_mode": (r"Payment\s*Mode\s*[:;]?\s*(Cheque|Bank\s*Transfer)",
                     lambda s: "Bank Transfer" if "bank" in s.lower() else "Cheque"),
    "unit_no": (r"Unit\s*No\.?\s*[:;]?\s*(\d+)", lambda s: s),
}


def extract_fields(text):
    out = {}
    for key, (pat, fn) in FIELD_PATTERNS.items():
        m = re.search(pat, text, re.I)
        try:
            out[key] = fn(m.group(1).strip()) if m else None
        except Exception:
            out[key] = None
    return out


# ----------------------------------------------------------------- metrics
def cer(pred, gold):
    import jiwer
    if not gold:
        return 0.0
    return jiwer.cer(gold, pred) if pred else 1.0


def match_clauses(pred, gold):
    """Pair each gold clause with a predicted clause, best-matching pairs first (global greedy).
    Matching gold clauses one by one in order let one merged/missing clause 'steal' its
    neighbour's match and cascade into several false misses."""
    sims = [(difflib.SequenceMatcher(None, g.lower(), p.lower()).ratio(), gi, pi)
            for gi, g in enumerate(gold) for pi, p in enumerate(pred)]
    sims.sort(reverse=True)
    g_to_p, used_p = {}, set()
    for sim, gi, pi in sims:
        if gi in g_to_p or pi in used_p:
            continue
        g_to_p[gi] = (pi, sim)
        used_p.add(pi)
    pairs = []
    for gi, g in enumerate(gold):
        if gi in g_to_p:
            pi, sim = g_to_p[gi]
            pairs.append((g, pred[pi], sim))
        else:
            pairs.append((g, "", 0.0))
    return pairs


def process(image_path, engine, preprocess=True):
    img = Image.open(image_path)
    t0 = time.time()
    if preprocess:
        img = remove_colored_ink(img)
    text = ENGINES[engine](img)
    secs = time.time() - t0
    return {"text": text, "clauses": split_clauses(text),
            "fields": extract_fields(text), "seconds": secs}


def evaluate(data_dir, engine, limit=None, kinds=("clean", "photo"), preprocess=True):
    labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.json")))[:limit]
    rows = []
    os.makedirs(os.path.join(data_dir, "ocr", engine), exist_ok=True)
    for lp in labels:
        gold = json.load(open(lp, encoding="utf-8"))
        g_clauses = [c["text"] for c in gold["clauses"]]
        for kind in kinds:
            img_rel = gold.get(f"image_{kind}")
            if not img_rel:
                continue
            res = process(os.path.join(data_dir, img_rel), engine, preprocess)
            with open(os.path.join(data_dir, "ocr", engine, f"{gold['contract_id']}_{kind}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            pairs = match_clauses(res["clauses"], g_clauses)
            clause_cer = sum(cer(p, g) for g, p, _ in pairs) / max(1, len(pairs))
            found = sum(1 for _, _, s in pairs if s >= 0.8)
            fields_ok = sum(1 for k in FIELD_PATTERNS
                            if res["fields"].get(k) == gold["fields"].get(k))
            rows.append({
                "contract_id": gold["contract_id"], "kind": kind, "engine": engine,
                "preprocess": preprocess,
                "gold_clauses": len(g_clauses), "pred_clauses": len(res["clauses"]),
                "clauses_found": found, "clause_cer": round(clause_cer, 4),
                "fields_correct": fields_ok, "fields_total": len(FIELD_PATTERNS),
                "seconds": round(res["seconds"], 2),
            })
            print(f"{gold['contract_id']} {kind:5s} clauses {found}/{len(g_clauses)} "
                  f"CER {clause_cer:.3f} fields {fields_ok}/{len(FIELD_PATTERNS)} "
                  f"({res['seconds']:.1f}s)")
    return rows


def rescore(data_dir, engine, kinds=("clean", "photo")):
    """Re-split and re-score saved OCR output (data_dir/ocr/<engine>/*.json) without running OCR again.
    Use after changing the clause splitter or the scoring."""
    rows = []
    for lp in sorted(glob.glob(os.path.join(data_dir, "labels", "*.json"))):
        gold = json.load(open(lp, encoding="utf-8"))
        g_clauses = [c["text"] for c in gold["clauses"]]
        for kind in kinds:
            op = os.path.join(data_dir, "ocr", engine, f"{gold['contract_id']}_{kind}.json")
            if not os.path.exists(op):
                continue
            res = json.load(open(op, encoding="utf-8"))
            res["clauses"] = split_clauses(res["text"])
            res["fields"] = extract_fields(res["text"])
            with open(op, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            pairs = match_clauses(res["clauses"], g_clauses)
            rows.append({
                "contract_id": gold["contract_id"], "kind": kind, "engine": engine, "preprocess": True,
                "gold_clauses": len(g_clauses), "pred_clauses": len(res["clauses"]),
                "clauses_found": sum(1 for _, _, sim in pairs if sim >= 0.8),
                "clause_cer": round(sum(cer(p, g) for g, p, _ in pairs) / max(1, len(pairs)), 4),
                "fields_correct": sum(1 for k in FIELD_PATTERNS if res["fields"].get(k) == gold["fields"].get(k)),
                "fields_total": len(FIELD_PATTERNS), "seconds": round(res.get("seconds", 0), 2),
            })
    for r in rows:
        if r["clauses_found"] < r["gold_clauses"]:
            print(f"{r['contract_id']} {r['kind']:5s} clauses {r['clauses_found']}/{r['gold_clauses']} CER {r['clause_cer']:.3f}")
    return rows


def summarize(rows):
    print("\n=== SUMMARY ===")
    for kind in ["clean", "photo"]:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        g = sum(r["gold_clauses"] for r in rs)
        f = sum(r["clauses_found"] for r in rs)
        fc = sum(r["fields_correct"] for r in rs)
        ft = sum(r["fields_total"] for r in rs)
        tag = rs[0]['engine'] + ("+prep" if rs[0].get("preprocess") else "")
        print(f"{tag:14s} {kind:5s} | clause recall {f}/{g} = {f / g:.1%} | "
              f"clause CER {sum(r['clause_cer'] for r in rs) / len(rs):.2%} | "
              f"field accuracy {fc}/{ft} = {fc / ft:.1%} | "
              f"{sum(r['seconds'] for r in rs) / len(rs):.1f}s/page")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out")
    ap.add_argument("--engine", choices=list(ENGINES), default="rapidocr")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--image", default=None, help="run on a single image and print the result")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score saved OCR output (after splitter/scoring changes) without running OCR")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="skip coloured-ink removal (to measure how much it helps)")
    args = ap.parse_args()

    if args.image:
        res = process(args.image, args.engine, not args.no_preprocess)
        print("---- FIELDS ----")
        for k, v in res["fields"].items():
            print(f"{k:18s} {v}")
        print("\n---- CLAUSES ----")
        for i, c in enumerate(res["clauses"], 1):
            print(f"{i}. {c}")
        return

    prep = not args.no_preprocess
    if args.rescore:
        rows = rescore(args.data, args.engine)
    else:
        rows = evaluate(args.data, args.engine, args.limit, preprocess=prep)
    out_csv = os.path.join(args.data, f"ocr_results_{args.engine}{'_prep' if prep else ''}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nDetails → {out_csv}  |  raw OCR → {args.data}/ocr/{args.engine}/")


if __name__ == "__main__":
    main()
