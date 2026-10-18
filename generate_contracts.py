"""
Renewal Shield — Phase 2: synthetic tenancy contract generator.

Reads the clause bank spreadsheet and produces fake tenancy contracts:
  out/clean/contract_0001.png    – clean scan-like page
  out/photo/contract_0001.png    – same page made to look like a phone photo
  out/labels/contract_0001.json  – ground truth: fields, clauses, labels, boxes
  out/manifest.csv               – one row per contract

Layout is a GENERIC tenancy contract, not a copy of the official Ejari/DLD form,
and every page carries a "SYNTHETIC SAMPLE" watermark so it can't be mistaken
for a real document.

Usage:
  python generate_contracts.py --bank renewal_shield_clause_bank_v0.3.xlsx --n 200
  python generate_contracts.py --n 10 --seed 7 --min-clauses 3 --max-clauses 6
"""

import argparse
import csv
import json
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

# ----------------------------------------------------------------- fake data
FIRST = ["Ahmed", "Fatima", "Rahul", "Priya", "Omar", "Sara", "John", "Maria",
         "Ali", "Aisha", "Vikram", "Noor", "Daniel", "Layla", "Hassan", "Elena",
         "Arjun", "Mariam", "Karim", "Sophie", "Yusuf", "Anjali", "Khalid", "Grace"]
LAST = ["Al Mansoori", "Khan", "Sharma", "Hassan", "Fernandes", "Al Hashimi",
        "Smith", "Patel", "Rahman", "Ivanova", "Mehta", "Al Suwaidi", "Ibrahim",
        "Nair", "Okafor", "Rossi", "Siddiqui", "Thomas", "Haddad", "Kapoor"]
COMPANIES = ["Al Noor Properties LLC", "Desert Rose Real Estate", "Gulf Crest Holdings",
             "Palm Gate Property Management", "Blue Dune Investments LLC"]
AREAS = ["Jumeirah Village Circle", "Dubai Marina", "Al Barsha", "International City",
         "Business Bay", "Deira", "Al Nahda", "Dubai Silicon Oasis", "Discovery Gardens",
         "Mirdif", "Jumeirah Lake Towers", "Al Qusais", "Arjan", "Motor City"]
BUILDING_WORDS = ["Tower", "Residence", "Heights", "Court", "Building", "Plaza"]
BUILDING_NAMES = ["Al Waha", "Marina View", "Sunrise", "Golden Sands", "Crystal",
                  "Al Zahra", "Lake Point", "Palm Breeze", "Silver Oak", "Royal"]
PROP_TYPES = [("Studio", 30000, 55000), ("1 BHK Apartment", 45000, 90000),
              ("2 BHK Apartment", 70000, 140000), ("3 BHK Apartment", 100000, 200000),
              ("Villa", 150000, 350000)]
PAYMENT = ["Cheque", "Cheque", "Cheque", "Bank Transfer"]
CHEQUES = [1, 2, 4, 6, 12]


def rand_name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def fake_fields(r):
    ptype, lo, hi = r.choice(PROP_TYPES)
    rent = int(r.randint(lo, hi) / 1000) * 1000
    start = date(2025, 1, 1) + timedelta(days=r.randint(0, 600))
    end = start + timedelta(days=364)
    contract_date = start - timedelta(days=r.randint(3, 30))
    furnished = r.random() < 0.3
    deposit = int(rent * (0.10 if furnished else 0.05))
    landlord = r.choice(COMPANIES) if r.random() < 0.4 else rand_name(r)
    payment_mode = r.choice(PAYMENT)
    # bank transfers have no cheques
    num_cheques = r.choice(CHEQUES) if payment_mode == "Cheque" else 0
    return {
        "contract_date": contract_date.isoformat(),
        "landlord_name": landlord,
        "tenant_name": rand_name(r),
        "property_type": ptype,
        "property_usage": "Residential",
        "furnished": "Yes" if furnished else "No",
        "building_name": f"{r.choice(BUILDING_NAMES)} {r.choice(BUILDING_WORDS)}",
        "unit_no": str(r.randint(101, 2405)),
        "area": r.choice(AREAS),
        "plot_no": f"{r.randint(100, 699)}-{r.randint(1000, 9999)}",
        "makani_no": f"{r.randint(10000, 99999)} {r.randint(10000, 99999)}",
        "premises_no": str(r.randint(100000000, 999999999)),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "annual_rent": rent,
        "contract_value": rent,
        "security_deposit": deposit,
        "payment_mode": payment_mode,
        "num_cheques": num_cheques,
    }


def fmt_date(iso):
    d = date.fromisoformat(iso)
    return d.strftime("%d/%m/%Y")


def fmt_aed(x):
    return f"AED {x:,.0f}"


# ----------------------------------------------------------------- clause bank
def load_bank(path):
    df = pd.read_excel(path, sheet_name="Clause Bank").fillna("")
    df["text"] = df.apply(
        lambda row: row["edited_text"].strip() if str(row["edited_text"]).strip()
        else row["clause_text"].strip(), axis=1)
    # drop rows the reviewer explicitly rejected
    df = df[df["verified"].astype(str).str.lower() != "no"].copy()
    return df


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fill_placeholders(text, fields, r):
    """Replace {deposit}, {rent}, ... with values from this contract's fields."""
    rent = fields["annual_rent"]
    deposit = fields["security_deposit"]
    values = {
        "rent": fmt_aed(rent),
        "deposit": fmt_aed(deposit),
        "rent_next_year": fmt_aed(round(rent * r.uniform(1.10, 1.25), -3)),
        "cleaning_fee": fmt_aed(max(500, round(deposit * r.uniform(0.4, 0.8), -2))),
    }
    return text.format_map(_Safe(values))


def pick_clauses(r, bank, k):
    """Pick k clauses from k different categories, ~50/50 problem vs fine."""
    cats = list(bank["category"].unique())
    r.shuffle(cats)
    chosen = []
    for cat in cats[:k]:
        sub = bank[bank["category"] == cat]
        want = "problem" if r.random() < 0.5 else "fine"
        pool = sub[sub["label"] == want]
        if pool.empty:
            pool = sub
        chosen.append(pool.sample(1, random_state=r.randint(0, 10**9)).iloc[0])
    return chosen


# ----------------------------------------------------------------- rendering
W, H = 1654, 2339          # A4 at 200 dpi
MARGIN = 110


def load_fonts(font_dir=None):
    candidates = {
        "regular": ["DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "times.ttf", "Times New Roman.ttf"],
        "bold": ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "timesbd.ttf", "Times New Roman Bold.ttf"],
    }
    dirs = [d for d in [font_dir,
                        "/usr/share/fonts/truetype/dejavu",
                        "/usr/share/fonts/truetype/liberation",
                        "C:/Windows/Fonts",
                        "/Library/Fonts", "/System/Library/Fonts/Supplemental"] if d]

    def find(kind, size):
        for d in dirs:
            for name in candidates[kind]:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return ImageFont.truetype(p, size)
        return ImageFont.load_default(size=size)

    return {
        "title": find("bold", 46), "h": find("bold", 30), "label": find("bold", 25),
        "body": find("regular", 25), "small": find("regular", 21), "wm": find("bold", 120),
    }


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(fields, clauses, fonts, r):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    boxes_fields, boxes_clauses = {}, []
    y = MARGIN

    # header
    title = "RESIDENTIAL TENANCY CONTRACT"
    tw = d.textlength(title, font=fonts["title"])
    d.text(((W - tw) / 2, y), title, font=fonts["title"], fill="black")
    y += 70
    sub = f"Contract Date: {fmt_date(fields['contract_date'])}"
    d.text((MARGIN, y), sub, font=fonts["body"], fill="black")
    boxes_fields["contract_date"] = list(d.textbbox((MARGIN, y), sub, font=fonts["body"]))
    y += 55
    d.line([(MARGIN, y), (W - MARGIN, y)], fill="black", width=3)
    y += 25

    def section(name):
        nonlocal y
        d.rectangle([MARGIN, y, W - MARGIN, y + 44], fill=(225, 225, 225))
        d.text((MARGIN + 12, y + 6), name, font=fonts["h"], fill="black")
        y += 62

    def row(pairs):
        """pairs: list of (label, field_key, display) laid out in 2 columns."""
        nonlocal y
        col_w = (W - 2 * MARGIN) / 2
        for i, (label, key, val) in enumerate(pairs):
            x = MARGIN + (i % 2) * col_w
            if i % 2 == 0 and i > 0:
                y += 44
            d.text((x, y), f"{label}:", font=fonts["label"], fill="black")
            lx = x + d.textlength(f"{label}: ", font=fonts["label"])
            d.text((lx, y), str(val), font=fonts["body"], fill="black")
            if key:
                boxes_fields[key] = list(d.textbbox((lx, y), str(val), font=fonts["body"]))
        y += 60

    section("1. PARTIES")
    row([("Landlord", "landlord_name", fields["landlord_name"]),
         ("Tenant", "tenant_name", fields["tenant_name"])])

    section("2. PROPERTY")
    row([("Property Type", "property_type", fields["property_type"]),
         ("Usage", "property_usage", fields["property_usage"]),
         ("Building", "building_name", fields["building_name"]),
         ("Unit No.", "unit_no", fields["unit_no"]),
         ("Area", "area", fields["area"]),
         ("Plot No.", "plot_no", fields["plot_no"]),
         ("Makani No.", "makani_no", fields["makani_no"]),
         ("Premises No.", "premises_no", fields["premises_no"]),
         ("Furnished", "furnished", fields["furnished"])])

    section("3. TERM AND PAYMENT")
    row([("Start Date", "start_date", fmt_date(fields["start_date"])),
         ("End Date", "end_date", fmt_date(fields["end_date"])),
         ("Annual Rent", "annual_rent", fmt_aed(fields["annual_rent"])),
         ("Contract Value", "contract_value", fmt_aed(fields["contract_value"])),
         ("Security Deposit", "security_deposit", fmt_aed(fields["security_deposit"])),
         ("Payment Mode", "payment_mode", fields["payment_mode"]),
         ("No. of Cheques", "num_cheques",
          fields["num_cheques"] if fields["num_cheques"] else "N/A")])

    section("4. ADDITIONAL TERMS")
    max_w = W - 2 * MARGIN - 50
    for i, c in enumerate(clauses, 1):
        lines = wrap(d, c["text"], fonts["body"], max_w)
        x0, y0 = MARGIN, y
        d.text((MARGIN, y), f"{i}.", font=fonts["body"], fill="black")
        for ln in lines:
            d.text((MARGIN + 45, y), ln, font=fonts["body"], fill="black")
            y += 36
        boxes_clauses.append([x0, y0, W - MARGIN, y])
        y += 18

    # signatures
    y = max(y + 40, H - 300)
    for i, who in enumerate(["Landlord Signature", "Tenant Signature"]):
        x = MARGIN + i * (W - 2 * MARGIN) / 2
        d.line([(x, y + 60), (x + 450, y + 60)], fill="black", width=2)
        d.text((x, y + 70), who, font=fonts["small"], fill="black")
        # squiggle "signature"
        pts, sx = [], x + 30
        for t in range(40):
            pts.append((sx + t * 9, y + 40 + r.randint(-14, 14)))
        d.line(pts, fill=(20, 30, 120), width=3)

    # footer + watermark
    d.text((MARGIN, H - 80), "SYNTHETIC SAMPLE — generated for Renewal Shield research. Not a real contract.",
           font=fonts["small"], fill=(110, 110, 110))
    wm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    text = "SYNTHETIC SAMPLE"
    tw = wd.textlength(text, font=fonts["wm"])
    wd.text(((W - tw) / 2, H / 2 - 60), text, font=fonts["wm"], fill=(200, 0, 0, 38))
    wm = wm.rotate(35, resample=Image.BICUBIC)
    img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
    return img, boxes_fields, boxes_clauses


# ----------------------------------------------------------------- photo look
def photo_augment(img, r):
    """Make a clean page look like a phone photo. Boxes refer to the clean image."""
    # paper tint + brightness/contrast
    img = ImageEnhance.Brightness(img).enhance(r.uniform(0.85, 1.05))
    img = ImageEnhance.Contrast(img).enhance(r.uniform(0.8, 1.1))
    tint = np.array(img).astype(np.float32)
    tint *= np.array([r.uniform(0.95, 1.0), r.uniform(0.93, 1.0), r.uniform(0.85, 0.98)])
    # uneven lighting / shadow gradient
    gx = np.linspace(r.uniform(0.75, 1.0), r.uniform(0.9, 1.05), img.width)
    gy = np.linspace(r.uniform(0.85, 1.05), r.uniform(0.75, 1.0), img.height)
    tint *= np.outer(gy, gx)[..., None]
    # sensor noise
    tint += np.random.default_rng(r.randint(0, 10**9)).normal(0, r.uniform(2, 7), tint.shape)
    img = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8))
    # slight rotation on a darker "table" background
    bg = tuple(r.randint(60, 140) for _ in range(3))
    img = img.rotate(r.uniform(-3, 3), resample=Image.BICUBIC, expand=True, fillcolor=bg)
    # blur
    if r.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.4, 1.6)))
    # downscale like a phone upload
    scale = r.uniform(0.55, 0.8)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    return img


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="renewal_shield_clause_bank_v0.3.xlsx")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-clauses", type=int, default=3)
    ap.add_argument("--max-clauses", type=int, default=6)
    ap.add_argument("--font-dir", default=None)
    ap.add_argument("--no-photo", action="store_true")
    args = ap.parse_args()

    r = random.Random(args.seed)
    bank = load_bank(args.bank)
    fonts = load_fonts(args.font_dir)
    for sub in ["clean", "photo", "labels"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    manifest = []
    for i in range(1, args.n + 1):
        cid = f"contract_{i:04d}"
        fields = fake_fields(r)
        k = r.randint(args.min_clauses, min(args.max_clauses, bank["category"].nunique()))
        picked = pick_clauses(r, bank, k)
        # cheque clauses only make sense if the tenant actually pays by multiple cheques
        if any("cheque" in str(p["text"]).lower() for p in picked) and (
                fields["payment_mode"] != "Cheque" or fields["num_cheques"] < 2):
            fields["payment_mode"] = "Cheque"
            fields["num_cheques"] = r.choice([2, 4, 6, 12])
        clauses = [{
            "clause_id": p["id"], "category": p["category"], "label": p["label"],
            "severity": p["severity"], "difficulty": p["difficulty"],
            "text": fill_placeholders(p["text"], fields, r),
            "law_ref": p["law_ref"],
            "label_verified": str(p["verified"]).lower() == "yes",
        } for p in picked]

        img, fbox, cbox = render(fields, clauses, fonts, r)
        for c, b in zip(clauses, cbox):
            c["bbox_clean"] = b
        img.save(os.path.join(args.out, "clean", f"{cid}.png"))
        if not args.no_photo:
            photo_augment(img, r).save(os.path.join(args.out, "photo", f"{cid}.png"))

        label = {
            "contract_id": cid,
            "synthetic": True,
            "image_clean": f"clean/{cid}.png",
            "image_photo": None if args.no_photo else f"photo/{cid}.png",
            "fields": fields,
            "field_bboxes_clean": fbox,
            "clauses": clauses,
            "has_problem": any(c["label"] == "problem" for c in clauses),
            "all_labels_verified": all(c["label_verified"] for c in clauses),
        }
        with open(os.path.join(args.out, "labels", f"{cid}.json"), "w", encoding="utf-8") as f:
            json.dump(label, f, indent=2, ensure_ascii=False)

        manifest.append({
            "contract_id": cid,
            "n_clauses": len(clauses),
            "n_problem": sum(c["label"] == "problem" for c in clauses),
            "n_hard": sum(c["difficulty"] == "hard" for c in clauses),
            "categories": "|".join(c["category"] for c in clauses),
            "clause_ids": "|".join(c["clause_id"] for c in clauses),
            "all_labels_verified": label["all_labels_verified"],
        })

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    total = sum(m["n_clauses"] for m in manifest)
    probs = sum(m["n_problem"] for m in manifest)
    print(f"Generated {args.n} contracts, {total} clauses ({probs} problem, {total - probs} fine) → {args.out}/")


if __name__ == "__main__":
    main()
