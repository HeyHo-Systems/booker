#!/usr/bin/env python
import os, argparse, io, requests, stripe, shutil
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pypdf import PdfWriter
from tqdm import tqdm      # nice progress bar
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

stripe.api_key = os.environ["STRIPE_KEY"]     # export STRIPE_KEY=sk_live_...

# ---------- helpers ----------------------------------------------------------
def month_window(year: int, month: int):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end   = start + relativedelta(months=1)
    return int(start.timestamp()), int(end.timestamp())

def list_monthly_payouts(year: int, month: int):
    start, end = month_window(year, month)
    return stripe.Payout.list(
        created={"gte": start, "lt": end},
        status="paid",
        limit=100
    ).auto_paging_iter()

def docs_for_payout(po_id: str):
    """Return a list of signed URLs for invoice + credit-note PDFs."""
    btc = stripe.BalanceTransaction.list(
        payout=po_id,
        expand=["data.source", "data.source.invoice", "data.source.invoice.credit_notes"],
        limit=100
    )
    urls = set()
    for tx in btc.auto_paging_iter():
        inv = getattr(tx.source, "invoice", None)
        if inv and inv.invoice_pdf:
            urls.add(inv.invoice_pdf)                # invoice itself
            if hasattr(inv, 'credit_notes') and inv.credit_notes:  # safely check if credit_notes exists
                for cn in inv.credit_notes:              # any credit-notes
                    if cn.get("pdf"):
                        urls.add(cn["pdf"])
    return sorted(urls)

def merge_pdfs(urls):
    writer = PdfWriter()
    for u in urls:
        pdf_bytes = io.BytesIO(requests.get(u, timeout=10).content)
        writer.append(pdf_bytes)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf

# ---------- main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year",  type=int)
    ap.add_argument("month", type=int, help="1-12")
    args = ap.parse_args()

    for p in tqdm(list_monthly_payouts(args.year, args.month), desc="payouts"):
        urls = docs_for_payout(p.id)
        if not urls:              # rare: zero-revenue payout
            continue
        pdf  = merge_pdfs(urls)
        fname = f"{datetime.utcfromtimestamp(p.arrival_date).date()}_" \
                f"{p.amount/100:.2f}{p.currency}_" \
                f"{p.id}.pdf"
        
        # Save directly to Downloads folder
        downloads_path = os.path.expanduser("~/Downloads")
        download_file_path = os.path.join(downloads_path, fname)
        
        with open(download_file_path, "wb") as f:
            f.write(pdf.read())
        
        tqdm.write(f"✔ saved {fname} to ~/Downloads")

if __name__ == "__main__":
    main()