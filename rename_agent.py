from __future__ import annotations
import argparse
import json
import os
import re
import sys
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
# ---------- LLM function schema ---------------
FUNCTION_SPEC = {
    "name": "extract_invoice_fields",
    "description": "Extract bookkeeping metadata from an invoice or receipt.",
    "parameters": {
        "type": "object",
        "properties": {
            "date":  {"type": "string", "description": "Date of the transaction in YYYY-MM-DD."},
            "method":{"type": "string", "description": "Payment method, e.g. visa, mastercard, sepa, cash."},
            "amount":{"type": "string", "description": "Total amount in the invoice, e.g. 123.45 EUR."},
            "usage": {"type": "string", "description": "Short lower_snake_case label describing purpose/vendor."}
        },
        "required": ["date", "method", "amount", "usage"]
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
            model="gpt-4",
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
    date = fields["date"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"Bad date: {date}")

    method = re.sub(r"[^a-z]", "", fields["method"].lower())

    amount = fields["amount"].upper().replace(",", ".").strip()
    amount = re.sub(r"\s+", "", amount)  # e.g. 123.45EUR

    usage = re.sub(r"[^a-z0-9_]", "_", fields["usage"].lower())
    usage = re.sub(r"_+", "_", usage).strip("_")[:30]

    return {"date": date, "method": method, "amount": amount, "usage": usage}

def build_filename(meta: Dict[str, str], original_suffix: str) -> str:
    date = meta["date"].replace("-", ".")
    return f"{date}_{meta['method']}_{meta['amount']}_{meta['usage']}{original_suffix}"

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
    new_path = path.with_name(new_name)
    if new_path.exists():
        print(f"⚠️  {new_name} already exists; skipping")
        return
    if dry_run:
        print(f"[DRY] {path.name}  ->  {new_name}")
    else:
        path.rename(new_path)
        print(f"✓ {path.name}  ->  {new_name}")
    if log_handle:
        log_handle.write(json.dumps({
            "old": str(path),
            "new": str(new_path),
            "fields": meta
        }) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Batch‑rename invoice files using OpenAI LLM.")
    parser.add_argument("folder", type=Path, help="Folder containing invoices")
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without changing files")
    parser.add_argument("--log", type=argparse.FileType("a"), help="Append JSONL log to this file")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print("Error: folder must be a directory", file=sys.stderr)
        sys.exit(1)

    for path in sorted(args.folder.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        process_file(path, dry_run=args.dry_run, log_handle=args.log)

if __name__ == "__main__":
    main()
