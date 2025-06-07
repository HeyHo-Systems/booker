"""
match_my_statements.py
---------------------
CLI tool that scans a credit‑card statement PDF (German Hanseatic layout),
matches each transaction line against a folder of *renamed* invoice files
(created by rename_my_invoices.py). This script identifies matching transactions
and generates a report.

Usage
-----
python match_my_statements.py --statement path/to/Statement.pdf \
                         --invoices  path/to/renamed_folder \
                         [--dry-run] [--fx-cache fx.json] [--debug]

Outputs
-------
- A JSON file with detailed matching data (Statement_results.json)
- A markdown report with tables of matches and unmatched items (Statement_report.md)

Both files are saved in the same directory as the input statement.

Dependencies
------------
pip install pdfplumber rapidfuzz openai requests python-dotenv numpy tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime, date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pdfplumber
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (uses env vars OPENAI_API_KEY etc.)
client = OpenAI()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INVOICE_RX = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})_(?P<method>[a-z0-9_]+)_(?P<amount>\d+\.\d{2})"
    r"(?P<ccy>[A-Z]{3})_(?P<slug>.+)"
)
EMBED_MODEL = "text-embedding-3-small"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_decimal(val: str) -> Decimal:
    val = val.replace(".", "").replace(",", ".").replace(" ", "")
    try:
        return Decimal(val)
    except InvalidOperation:
        return Decimal("0")


def parse_date_de(val: str) -> Optional[date]:
    try:
        return datetime.strptime(val, "%d.%m.%Y").date()
    except ValueError:
        return None


def load_fx_rate(usd_date: date, cache_path: Optional[Path]) -> Decimal:
    """
    Returns the USD→EUR mid‑market rate for the given date.
    Uses exchangerate.host and caches daily results in JSON.
    """
    if not cache_path:
        cache_path = Path(".fx_cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    key = usd_date.isoformat()
    if key in cache:
        return Decimal(str(cache[key]))
    url = f"https://api.exchangerate.host/{key}?base=USD&symbols=EUR"
    try:
        res = requests.get(url, timeout=10).json()
        rate = Decimal(str(res["rates"]["EUR"]))
        cache[key] = str(rate)
        cache_path.write_text(json.dumps(cache, indent=2))
        return rate
    except Exception:
        # Fallback to 1.0 to avoid crashing; will harm matching accuracy.
        return Decimal("1")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# -----------------------------------------------------------------------------
# Invoice side
# -----------------------------------------------------------------------------
def load_invoices(folder: Path) -> List[Dict]:
    rows = []
    for p in folder.glob("*.[pP][dDnN][fFgG]"):  # Match both .pdf and .png (case insensitive)
        m = INVOICE_RX.match(p.stem)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "path": p,
                "date": datetime.strptime(d["date"], "%y.%m.%d").date(),
                "amount": Decimal(d["amount"]),
                "ccy": d["ccy"],
                "slug": d["slug"].replace("_", " ").lower(),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Statement side
# -----------------------------------------------------------------------------
def extract_statement_rows(pdf_path: Path) -> List[Dict]:
    rows = []
    with pdfplumber.open(str(pdf_path)) as doc:
        for page_idx, page in enumerate(doc.pages):
            # crude: look at every text line
            text = page.extract_text(x_tolerance=1, y_tolerance=3)
            if not text:
                continue
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                # match booking date at line start
                m = re.match(r"(\d{2}\.\d{2}\.\d{4})\s+(\d{2}\.\d{2}\.\d{4})\s+Kartenumsatz\s+9174\s+(-?\d+[,\.]\d{2})", line)
                if not m:
                    i += 1
                    continue
                
                book_dt, trans_dt, amount_eur = m.groups()
                
                # Get merchant name from next line if available
                merchant = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Skip if next line looks like a header or footer
                    if not any(x in next_line.lower() for x in ["buchungs", "beschreibung", "datum", "hanseatic", "übertrag", "saldo"]):
                        merchant = next_line.strip()
                        i += 1  # Skip the merchant line in next iteration
                
                rows.append(
                    {
                        "page": page_idx,
                        "book_dt": parse_date_de(book_dt),
                        "trans_dt": parse_date_de(trans_dt),
                        "descr": merchant.lower() if merchant else "kartenumsatz 9174",
                        "eur": parse_decimal(amount_eur).quantize(Decimal("0.01"), ROUND_HALF_UP),
                    }
                )
                i += 1
    return rows


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """
    Returns the embedding vector for the given text using OpenAI >=1.0 client.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text[:200]])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def get_merchant_context(descr: str) -> str:
    """
    Analyze merchant description to provide relevant context for matching.
    Returns a string with known patterns and variations.
    """
    descr = descr.lower()
    contexts = []
    
    # Common SaaS/Cloud services patterns
    if any(x in descr for x in ['cloud', 'api', 'hosting', 'subscription']):
        contexts.append("This is a cloud/SaaS service which often has variations in billing descriptions.")
        contexts.append("Monthly charges may vary slightly due to usage-based pricing.")
        
    # Payment processors and financial services
    if any(x in descr for x in ['payment', 'stripe', 'paypal', 'billing']):
        contexts.append("This is a payment processor which may show different merchant names for the same service.")
        
    # Development tools and platforms
    if any(x in descr for x in ['github', 'gitlab', 'vercel', 'heroku', 'twilio', 'openai']):
        contexts.append("This is a development platform/tool with potential variations in product names.")
        contexts.append("Charges might include product name, subscription tier, or usage period.")
        
    # Fuel and travel
    if any(x in descr for x in ['shell', 'bp', 'fuel', 'gas', 'tankstelle']):
        contexts.append("This is a fuel/gas station purchase.")
        contexts.append("Station numbers and location details may vary in descriptions.")
        
    # Default context
    if not contexts:
        contexts.append("This is a general merchant transaction.")
        
    return " ".join(contexts)


def referee_match(stmt: Dict, inv: Dict) -> Tuple[bool, float]:
    """
    Ask GPT‑4‑turbo whether the statement line and invoice describe the same purchase.
    Returns (match_boolean, confidence_float).
    Enhanced with merchant context and common variations.
    """
    merchant_context = get_merchant_context(stmt['descr'])
    
    prompt_system = (
        "You are a bookkeeping assistant specializing in matching credit card statements "
        "to invoices. Consider these key factors:\n"
        "1. Merchant names may vary between statement and invoice\n"
        "2. Amounts may differ slightly due to FX rates\n"
        "3. Dates should typically be within 2 weeks\n"
        "4. Transaction types and merchant patterns matter\n"
        "Return ONLY valid JSON like {\"match\":true, \"confidence\":0.83}"
    )
    
    prompt_user = (
        f"STATEMENT:\n"
        f"booking_date={stmt['book_dt']}\n"
        f"trans_date={stmt['trans_dt']}\n"
        f"description=\"{stmt['descr']}\"\n"
        f"amount_eur={stmt['eur']}\n\n"
        f"INVOICE:\n"
        f"file=\"{inv['path'].name}\"\n"
        f"date={inv['date']}\n"
        f"amount={inv['amount']} {inv['ccy'].upper()}\n\n"
        f"MERCHANT CONTEXT:\n{merchant_context}"
    )
    
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        temperature=0,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("match")), float(data.get("confidence", 0))
    except Exception:
        return False, 0.0


def slugify(text: str, cache: Dict[str, str]) -> str:
    if text in cache:
        return cache[text]
    # heuristic quick slug
    s = re.sub(r"[^a-z0-9]+", " ", text).strip()
    cache[text] = s
    return s


# Clean description for merchant name extraction
def clean_descr(raw: str) -> str:
    """
    Remove boiler‑plate tokens like 'kartenumsatz', card suffix, etc.,
    so that the remaining text is mostly the merchant name.
    """
    trash = {"kartenumsatz", "gutschrift", "visa", "mastercard", "debit", "credit"}
    tokens = re.split(r"[^a-z0-9]+", raw.lower())
    filtered = [t for t in tokens if t and t not in trash and not t.isdigit()]
    return " ".join(filtered)


# -----------------------------------------------------------------------------
# Matching logic
# -----------------------------------------------------------------------------
def normalize_merchant_name(name: str) -> str:
    """
    Normalize merchant names for better matching by:
    - Removing common suffixes (.com, inc, gmbh)
    - Removing location info after commas
    - Removing special characters
    - Converting to lowercase
    - Normalizing common terms (e.g. fuel station variants)
    """
    # Convert to lowercase and remove special chars
    name = name.lower()
    
    # Remove location info after comma
    name = name.split(',')[0]
    
    # Remove common suffixes
    suffixes = ['.com', '.co', 'inc', 'gmbh', 'ltd', 'llc']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Normalize common terms
    fuel_terms = {
        'oil': 'fuel',
        'gas': 'fuel',
        'tankstelle': 'fuel',
        'station': 'fuel',
        'fuel': 'fuel', 
        'shell': 'fuel',
        'bp': 'fuel',
        'aral': 'fuel',
        'esso': 'fuel',
        'total': 'fuel',
        'star': 'fuel',
        'orlen': 'fuel',
        'jet': 'fuel',
        'avia': 'fuel',
        'hem': 'fuel',
        'agip': 'fuel'
    }
    
    words = name.split()
    normalized = []
    for word in words:
        # Skip pure numbers (like "209")
        if word.isdigit():
            continue
        # Replace with normalized term if exists
        normalized.append(fuel_terms.get(word, word))
    
    # Remove special characters and normalize spaces
    name = ' '.join(normalized)
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def amount_similarity(stmt_amount: Decimal, inv_amount: Decimal, inv_currency: str, fx_rate: Decimal) -> float:
    """
    Compare amounts with FX rate tolerance
    Returns a score between 0 and 1
    Credit card statements often show expenses as negative, so we compare absolute values
    """
    # Use absolute values for comparison since credit card statements show expenses as negative
    stmt_abs = abs(stmt_amount)
    inv_abs = abs(inv_amount)
    
    # Check for exact match first with a slight tolerance for rounding
    if abs(stmt_abs - inv_abs) < Decimal('0.10') and inv_currency == 'EUR':
        return 1.0
        
    if inv_currency != 'EUR':
        # Allow for FX rate variations of ±10%
        base_rate = fx_rate
        min_rate = base_rate * Decimal('0.90')
        max_rate = base_rate * Decimal('1.10')
        
        min_eur = inv_abs * min_rate
        max_eur = inv_abs * max_rate
        
        # If statement amount falls within the range, it's a perfect match
        if min_eur <= stmt_abs <= max_eur:
            return 1.0
            
        # Otherwise calculate relative difference using closest value
        closest = min(abs(stmt_abs - min_eur), abs(stmt_abs - max_eur))
        rel_diff = closest / max(stmt_abs, max_eur)
    else:
        rel_diff = abs(stmt_abs - inv_abs) / max(stmt_abs, inv_abs)
    
    # Convert difference to similarity score
    return float(max(0, 1 - min(rel_diff, 1)))

def match_rows(
    stmt_rows: List[Dict],
    inv_rows: List[Dict],
    fx_cache: Path,
    threshold: float = 0.5,
    debug: bool = False,
) -> Dict[int, int]:
    """
    Returns a dict mapping statement index → invoice index
    """
    # preprocess statement rows: clean merchant text and embed
    valid_stmt_rows = []
    for idx, r in enumerate(stmt_rows):
        cleaned = clean_descr(r["descr"])
        if debug:
            print(f"\nRaw description for row {idx}: \"{r['descr']}\"")
            print(f"Cleaned description: \"{cleaned}\"")
        if not cleaned:
            if debug:
                print(f"⚠️  Statement row {idx} cleaned to empty string; skipping")
            continue
        r["slug"] = normalize_merchant_name(cleaned)
        if debug:
            print(f"Normalized description: \"{r['slug']}\"")
        r["vec"] = get_embedding(r["slug"])
        valid_stmt_rows.append(r)

    # if some rows were skipped, update stmt_rows reference
    stmt_rows = valid_stmt_rows
    for i in inv_rows:
        i["slug"] = normalize_merchant_name(i["slug"].replace("_", " "))
        embed_text = f"{i['slug']} {i['amount']} {i['ccy']}"
        i["vec"] = get_embedding(embed_text)

    matches: Dict[int, int] = {}
    used_invoices = set()

    if not stmt_rows:
        return matches  # nothing to match
    stmt_mat = np.stack([r["vec"] for r in stmt_rows])
    inv_mat = np.stack([i["vec"] for i in inv_rows]).T  # transpose for dot product
    denom_stmt = np.linalg.norm(stmt_mat, axis=1, keepdims=True)
    denom_inv = np.linalg.norm(inv_mat, axis=0, keepdims=True)
    sims = stmt_mat @ inv_mat / (denom_stmt * denom_inv + 1e-9)  # shape (S, I)

    for s_idx, s in enumerate(tqdm(stmt_rows, desc="⚖️  Matching rows")):
        # get top 6 invoice indices by cosine similarity
        top_idx = np.argsort(-sims[s_idx])[:6]
        if debug:
            print(f"\n🔎 Statement row {s_idx}: \"{s['descr'][:60]}…\" "
                  f"EUR {s['eur']}  ({s['trans_dt']})")
        
        # Store all potential matches for this statement row
        potential_matches = []
        
        for i_idx in top_idx:
            if i_idx in used_invoices:
                continue
            inv = inv_rows[i_idx]

            # quick heuristic score
            sim = float(sims[s_idx, i_idx])
            date_delta = abs((s["trans_dt"] - inv["date"]).days)
            date_score = 1 - min(date_delta, 14) / 14
            
            try:
                amt_score = amount_similarity(
                    s["eur"], 
                    inv["amount"], 
                    inv["ccy"],
                    load_fx_rate(inv["date"], fx_cache)
                )
            except Exception:
                amt_score = 0.0
            
            # Perfect matches on date and amount should boost the overall score
            perfect_match = date_delta == 0 and amt_score > 0.95
            
            # Adjust weights based on similarity and perfect matches
            if perfect_match:  # Perfect date and amount - be more lenient on name
                heuristic = 0.3 * sim + 0.5 * amt_score + 0.2 * date_score + 0.2  # Bonus for perfect match
            elif sim > 0.6:  # High name similarity - be more lenient on other factors
                heuristic = 0.6 * sim + 0.25 * amt_score + 0.15 * date_score
            else:  # Normal weighting with more weight on amount
                heuristic = 0.4 * sim + 0.4 * amt_score + 0.2 * date_score

            if debug:
                print(f"   → Candidate {inv['path'].name:<40} "
                      f"sim={sim:.2f} dateΔ={date_delta}d amtΔ={(1-amt_score):.2%} "
                      f"heuristic={heuristic:.2f}", end="")

            # Use LLM referee more often:
            # 1. Always for high similarity but low heuristic
            # 2. For medium similarity with decent heuristic
            # 3. For borderline cases
            should_use_llm = (
                (sim > 0.6 and heuristic < threshold) or  # High sim but other factors off
                (sim > 0.4 and heuristic > 0.35) or      # Medium sim with decent score
                (0.45 <= heuristic <= 0.55)              # Borderline cases
            )

            if heuristic >= threshold:
                match, conf = True, heuristic
            elif should_use_llm:
                match, conf = referee_match(s, inv)
            else:
                match, conf = False, heuristic

            if debug and match:
                print(f"  ✅ accepted (conf={conf:.2f})")
            elif debug:
                print("  ✖")

            if match and conf >= 0.5:
                potential_matches.append((i_idx, conf, date_delta, sim))  # Added sim as 4th element

        # If we have multiple matches, prioritize semantic similarity only for very clear matches
        if potential_matches:
            # Check for very high semantic similarity with significant gap
            high_sim_matches = [m for m in potential_matches if m[3] > 0.6]  # Lowered threshold
            
            if high_sim_matches and len(potential_matches) > 1:
                # Only prioritize semantic similarity if there's a significant gap
                max_sim = max(m[3] for m in potential_matches)
                non_high_sim_matches = [m for m in potential_matches if m[3] <= 0.6]
                max_non_high_sim = max((m[3] for m in non_high_sim_matches), default=0)
                
                # Only prioritize if the top semantic match is significantly better than non-semantic matches
                if max_sim > 0.6 and (max_sim - max_non_high_sim) > 0.1:
                    high_sim_matches.sort(key=lambda x: (-x[3], -x[1], x[2]))  # -sim, -conf, date
                    best_idx = high_sim_matches[0][0]
                else:
                    # Fall back to original logic (confidence first)
                    potential_matches.sort(key=lambda x: (-x[1], x[2]))  # -conf, date
                    best_idx = potential_matches[0][0]
            else:
                # For single matches or no high similarity, use original logic
                potential_matches.sort(key=lambda x: (-x[1], x[2]))  # -conf, date
                best_idx = potential_matches[0][0]
                
            matches[s_idx] = best_idx
            used_invoices.add(best_idx)

    return matches


def generate_markdown_report(
    stmt_rows: List[Dict],
    inv_rows: List[Dict],
    matches: Dict[int, int],
    output_path: Path
) -> None:
    """Generate a markdown report with matches overview and tables."""
    
    # Calculate stats and unmatched items
    matched_stmt_indices = set(int(k) for k in matches.keys())
    matched_inv_indices = set(int(v) for v in matches.values())
    
    unmatched_stmt_indices = set(range(len(stmt_rows))) - matched_stmt_indices
    unmatched_inv_indices = set(range(len(inv_rows))) - matched_inv_indices
    
    # Start building the markdown content
    lines = [
        "# Statement Matching Report\n",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Overview\n",
        f"- Total statement rows: {len(stmt_rows)}",
        f"- Total invoice files: {len(inv_rows)}",
        f"- Matched pairs: {len(matches)}",
        f"- Unmatched statement rows: {len(unmatched_stmt_indices)}",
        f"- Unmatched invoice files: {len(unmatched_inv_indices)}\n",
        "## Matched Pairs\n",
        "| Statement Date | Description | Amount (EUR) | Invoice File |",
        "|---------------|-------------|--------------|--------------|"
    ]
    
    # Add matched pairs - sorted by statement date
    matched_pairs = []
    for stmt_idx, inv_idx in matches.items():
        stmt = stmt_rows[stmt_idx]
        inv = inv_rows[inv_idx]
        matched_pairs.append((stmt['trans_dt'], stmt, inv))
    
    # Sort matched pairs by statement date
    matched_pairs.sort(key=lambda x: x[0])
    
    for trans_dt, stmt, inv in matched_pairs:
        lines.append(
            f"| {trans_dt.strftime('%Y-%m-%d')} "
            f"| {stmt['descr'][:50]} "
            f"| {stmt['eur']} "
            f"| {Path(inv['path']).name} |"
        )
    
    # Add unmatched statement rows section - sorted by date
    lines.extend([
        "\n## Unmatched Statement Rows\n",
        "| Date | Description | Amount (EUR) |",
        "|------|-------------|--------------|"
    ])
    
    # Sort unmatched statement rows by date
    unmatched_stmt_sorted = sorted(unmatched_stmt_indices, key=lambda idx: stmt_rows[idx]['trans_dt'])
    for idx in unmatched_stmt_sorted:
        stmt = stmt_rows[idx]
        lines.append(
            f"| {stmt['trans_dt'].strftime('%Y-%m-%d')} "
            f"| {stmt['descr'][:50]} "
            f"| {stmt['eur']} |"
        )
    
    # Add unmatched invoice files section - sorted by date
    lines.extend([
        "\n## Unmatched Invoice Files\n",
        "| Date | Amount | Currency | Filename |",
        "|------|---------|----------|-----------|"
    ])
    
    # Sort unmatched invoices by date
    unmatched_inv_sorted = sorted(unmatched_inv_indices, key=lambda idx: inv_rows[idx]['date'])
    for idx in unmatched_inv_sorted:
        inv = inv_rows[idx]
        lines.append(
            f"| {inv['date'].strftime('%Y-%m-%d')} "
            f"| {inv['amount']} "
            f"| {inv['ccy']} "
            f"| {Path(inv['path']).name} |"
        )
    
    # Write the report
    output_path.write_text('\n'.join(lines))
    print(f"📝 Generated markdown report at {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Match statement rows with invoices.")
    
    # Add test mode argument first to check it early
    parser.add_argument("--test", action="store_true", help="Test report generation with dummy data")
    
    # Make these arguments not required if in test mode
    parser.add_argument("--statement", type=Path, required=False, help="Path to statement PDF")
    parser.add_argument("--invoices", type=Path, required=False, help="Path to invoices directory")
    parser.add_argument("--out", type=Path, required=False, help="Path for JSON output (optional, defaults to statement_name_results.json)")
    parser.add_argument("--report", type=Path, required=False, help="Path for markdown report output (optional, defaults to statement_name_report.md)")
    parser.add_argument("--fx-cache", type=Path, default=Path(".fx_rates.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--debug", action="store_true", help="Verbose matching output")
    args = parser.parse_args()

    if args.test:
        # For test mode, require the out path
        if not args.out:
            args.out = Path("test_results.json")
        if not args.report:
            args.report = Path("test_report.md")
            
        # Generate test data
        print("🧪 Testing report generation with dummy data...")
        test_stmt_rows = [
            {
                "page": 0,
                "book_dt": date(2025, 4, 1),
                "trans_dt": date(2025, 4, 1),
                "descr": "Test transaction 1",
                "eur": Decimal("123.45")
            },
            {
                "page": 0,
                "book_dt": date(2025, 4, 2),
                "trans_dt": date(2025, 4, 2),
                "descr": "Test transaction 2",
                "eur": Decimal("67.89")
            }
        ]
        test_inv_rows = [
            {
                "path": Path("test1.pdf"),
                "date": date(2025, 4, 1),
                "amount": Decimal("123.45"),
                "ccy": "EUR",
                "slug": "test1"
            },
            {
                "path": Path("test2.pdf"),
                "date": date(2025, 4, 3),
                "amount": Decimal("99.99"),
                "ccy": "USD",
                "slug": "test2"
            }
        ]
        test_matches = {0: 0}  # Match first statement to first invoice
        
        # Test JSON serialization
        results = {
            "matches": {},
            "statement_rows": [],
            "invoice_rows": []
        }
        
        # Process statement rows
        for idx, row in enumerate(test_stmt_rows):
            clean_row = {k: v for k, v in row.items() if k != "vec"}
            if "book_dt" in clean_row and clean_row["book_dt"]:
                clean_row["book_dt"] = clean_row["book_dt"].isoformat()
            if "trans_dt" in clean_row and clean_row["trans_dt"]:
                clean_row["trans_dt"] = clean_row["trans_dt"].isoformat()
            if "eur" in clean_row:
                clean_row["eur"] = str(clean_row["eur"])
            results["statement_rows"].append(clean_row)
            
        # Process invoice rows
        for idx, row in enumerate(test_inv_rows):
            clean_row = {k: v for k, v in row.items() if k != "vec"}
            if "path" in clean_row:
                clean_row["path"] = str(clean_row["path"])
            if "date" in clean_row and clean_row["date"]:
                clean_row["date"] = clean_row["date"].isoformat()
            if "amount" in clean_row:
                clean_row["amount"] = str(clean_row["amount"])
            results["invoice_rows"].append(clean_row)
        
        # Convert match indices to strings
        for s_idx, i_idx in test_matches.items():
            results["matches"][str(s_idx)] = int(i_idx)  # Convert numpy types to Python native types
            
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved test results to {args.out}")
        
        generate_markdown_report(test_stmt_rows, test_inv_rows, test_matches, args.report)
        return

    # In non-test mode, validate required arguments
    if not args.statement:
        sys.exit("--statement is required when not in test mode")
    if not args.invoices:
        sys.exit("--invoices is required when not in test mode")
    if not args.statement.exists():
        sys.exit("Statement PDF not found.")
    if not args.invoices.exists():
        sys.exit("Invoices folder not found.")
        
    # Automatically generate output filenames based on statement name
    statement_stem = args.statement.stem  # Gets filename without extension
    statement_dir = args.statement.parent
    
    if not args.out:
        args.out = statement_dir / f"{statement_stem}_results.json"
    
    if not args.report:
        args.report = statement_dir / f"{statement_stem}_report.md"

    print("📄 Extracting statement rows …")
    stmt_rows = extract_statement_rows(args.statement)
    print(f"   found {len(stmt_rows)} rows")

    print("📂 Loading invoices …")
    inv_rows = load_invoices(args.invoices)
    print(f"   found {len(inv_rows)} invoices")

    if not stmt_rows or not inv_rows:
        sys.exit("Nothing to match; aborting.")

    print("🔍 Matching …")
    matches = match_rows(stmt_rows, inv_rows, args.fx_cache, threshold=args.threshold, debug=args.debug)
    print(f"   matched {len(matches)}/{len(stmt_rows)} rows")
    
    if not args.dry_run:
        # Save JSON results
        results = {
            "matches": {},
            "statement_rows": [],
            "invoice_rows": []
        }
        
        # Process statement rows
        for idx, row in enumerate(stmt_rows):
            clean_row = {k: v for k, v in row.items() if k != "vec"}
            if "book_dt" in clean_row and clean_row["book_dt"]:
                clean_row["book_dt"] = clean_row["book_dt"].isoformat()
            if "trans_dt" in clean_row and clean_row["trans_dt"]:
                clean_row["trans_dt"] = clean_row["trans_dt"].isoformat()
            if "eur" in clean_row:
                clean_row["eur"] = str(clean_row["eur"])
            results["statement_rows"].append(clean_row)
            
        # Process invoice rows
        for idx, row in enumerate(inv_rows):
            clean_row = {k: v for k, v in row.items() if k != "vec"}
            if "path" in clean_row:
                clean_row["path"] = str(clean_row["path"])
            if "date" in clean_row and clean_row["date"]:
                clean_row["date"] = clean_row["date"].isoformat()
            if "amount" in clean_row:
                clean_row["amount"] = str(clean_row["amount"])
            results["invoice_rows"].append(clean_row)
        
        # Convert match indices to strings, ensuring Python native types
        for s_idx, i_idx in matches.items():
            results["matches"][str(int(s_idx))] = int(i_idx)  # Convert numpy types to Python native types
            
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved match results to {args.out}")
        
        # Always generate the report
        generate_markdown_report(stmt_rows, inv_rows, matches, args.report)
    else:
        print(f"[DRY] Would save match results to {args.out}")
        print(f"[DRY] Would generate markdown report to {args.report}")


if __name__ == "__main__":
    main()
