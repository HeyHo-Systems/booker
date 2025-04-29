from __future__ import annotations
import argparse
import json
import os
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract

# Load environment variables from .env file
load_dotenv()

# -------------- configuration -----------------
MAX_PROMPT_CHARS = 3500  # first N chars of document text sent to the LLM

# Directory structure
RAW_DIR = Path.home() / "Downloads" / "raw"
INVOICES_DIR = Path.home() / "Downloads" / "invoices"
PROCESSED_DIR = Path.home() / "Downloads" / "processed"

def ensure_directories():
    """Create the required directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INVOICES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LLM function schema ---------------
FUNCTION_SPEC = {
    "name": "extract_invoice_fields",
    "description": "Extract bookkeeping metadata from an invoice or receipt.",
    "parameters": {
        "type": "object",
        "properties": {
            "date":  {"type": "string", "description": "Date of the transaction in YYYY-MM-DD format."},
            "method":{"type": "string", "description": "Payment method with details. For PayPal extract the FULL email address (e.g. 'quantengoo@gmail.com'). For credit cards or bank transfers include the last 4 digits (e.g. 'visa 1234', 'mastercard 5678', 'sepa DE89'). If no digits available use: visa, mastercard, sepa, cash. If method cannot be determined, use 'unknown'."},
            "amount":{"type": "string", "description": "Total amount in the invoice including currency (e.g. 123.45 EUR, 50.00 USD)."},
            "purpose": {"type": "string", "description": "Description with company name if available, followed by short purpose (max 8 chars). Examples: 'vercel_hosting', 'openai_api', 'amazon_books'."}
        },
        "required": ["date", "method", "amount", "purpose"]
    },
}

# -------------- helpers ------------------------

def get_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return extract_text(str(path))
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            img = Image.open(path)
            return pytesseract.image_to_string(img)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        print(f"⚠️  Failed to extract text from {path.name}: {e}", file=sys.stderr)
        return ""

def call_llm(prompt: str) -> Optional[Dict[str, str]]:
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an accurate bookkeeping assistant."},
                {"role": "user", "content": prompt}
            ],
            tools=[{
                "type": "function",
                "function": FUNCTION_SPEC
            }],
            tool_choice={"type": "function", "function": {"name": "extract_invoice_fields"}},
            temperature=0.0,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {str(e)}", file=sys.stderr)
        print(f"Debug info - API key format: {'sk-...' if os.getenv('OPENAI_API_KEY', '').startswith('sk-') else 'invalid'}", file=sys.stderr)
        return None

def normalise(fields: Dict[str, str]) -> Dict[str, str]:
    # Convert date from YYYY-MM-DD to YY-MM-DD
    date = fields["date"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"Bad date: {date}")
    year, month, day = date.split("-")
    date = f"{year[-2:]}-{month}-{day}"  # Convert to YY-MM-DD

    # Handle payment method
    method = fields["method"].lower().strip()
    if not method:  # Empty or None
        method = "unknown"
    elif "@" in method:  # PayPal email address
        # Extract everything before the @ and clean it
        email_user = method.split("@")[0].strip()
        # Remove 'paypal' prefix if it exists in the email username
        email_user = re.sub(r'^paypal[_\s-]*', '', email_user)
        method = f"paypal_{email_user}"
    else:
        # Extract any 4-digit sequence that might be card/account numbers
        digits_match = re.search(r'[0-9x]{4,}', method)
        last_four = None
        if digits_match:
            digits = digits_match.group()
            last_four = digits[-4:]  # Get last 4 digits
            if 'x' not in last_four and last_four.isdigit():  # Only use if they're actual digits
                # Remove all non-letters from the method name
                base_method = re.sub(r'[^a-z]', '', method.split()[0])
                if base_method:
                    method = f"{base_method}_{last_four}"
                else:
                    method = "unknown"
            else:
                last_four = None
                
        if not last_four:  # If no valid digits found
            # Remove all non-letters and use base method
            cleaned_method = re.sub(r'[^a-z]', '', method)
            method = cleaned_method if cleaned_method else "unknown"

    # Format amount
    amount = fields["amount"].upper().replace(",", ".").strip()
    amount = re.sub(r"\s+", "", amount)  # e.g. 123.45EUR

    # Format purpose: company_shortpurpose (max 8 chars for short purpose)
    purpose = fields["purpose"].lower().strip()
    parts = purpose.split("_", 1)  # Split into company and rest
    if len(parts) > 1:
        company, rest = parts
        rest = rest[:8]  # Limit rest to 8 chars
        purpose = f"{company}_{rest}"
    else:
        purpose = purpose[:8]  # If no company, limit whole purpose to 8 chars
    
    # Clean up purpose
    purpose = re.sub(r"[^a-z0-9_]", "_", purpose)
    purpose = re.sub(r"_+", "_", purpose).strip("_")

    return {"date": date, "method": method, "amount": amount, "purpose": purpose}

def build_filename(meta: Dict[str, str], original_suffix: str) -> str:
    date = meta["date"].replace("-", ".")
    return f"{date}_{meta['method']}_{meta['amount']}_{meta['purpose']}{original_suffix}"

def process_file(path: Path, dry_run: bool=False, log_handle=None):
    text = get_text_from_file(path)[:MAX_PROMPT_CHARS]
    if not text.strip():
        print(f"⚠️  No text extracted from {path.name}; skipping")
        return
    fields = call_llm(text)
    if not fields:
        return
    try:
        meta = normalise(fields)
    except ValueError as e:
        print(f"⚠️  Validation error for {path.name}: {e}")
        return

    new_name = build_filename(meta, path.suffix.lower())
    new_path = INVOICES_DIR / new_name
    processed_path = PROCESSED_DIR / path.name

    if new_path.exists():
        print(f"⚠️  {new_name} already exists in invoices directory; skipping")
        return
    if processed_path.exists():
        print(f"⚠️  {path.name} already exists in processed directory; skipping")
        return

    if dry_run:
        print(f"[DRY] Would copy {path.name} -> {new_path}")
    else:
        try:
            # First copy to invoices directory with new name
            shutil.copy2(path, new_path)
            # Then move original to processed directory
            shutil.move(path, processed_path)
            print(f"✓ Created {new_path.name}")
            print(f"✓ Moved original to {processed_path.name}")
        except Exception as e:
            print(f"⚠️  Error processing {path.name}: {e}")
            # Cleanup if needed
            if new_path.exists():
                new_path.unlink()
            return

    if log_handle:
        log_handle.write(json.dumps({
            "original": str(path),
            "renamed": str(new_path),
            "processed": str(processed_path),
            "fields": meta
        }) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Batch‑rename invoice files using OpenAI LLM.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--log", type=argparse.FileType("a"), help="Append JSONL log to this file")
    args = parser.parse_args()

    # Ensure all required directories exist
    ensure_directories()

    if not RAW_DIR.exists() or not RAW_DIR.is_dir():
        print(f"Error: Raw directory {RAW_DIR} must exist and be a directory", file=sys.stderr)
        sys.exit(1)

    for path in sorted(RAW_DIR.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        process_file(path, dry_run=args.dry_run, log_handle=args.log)

if __name__ == "__main__":
    main()
