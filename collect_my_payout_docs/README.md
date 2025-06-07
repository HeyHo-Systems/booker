# collect_my_payout_docs

A CLI tool that collects and combines PDF invoices and credit notes for all Stripe payouts in a specified month, organizing them into a standard format for bookkeeping.

## Features

- **One file per Stripe payout**: Combines all invoices and credit notes into a single PDF
- **Complete documentation**: Includes all customer invoices and refund credit notes
- **Efficient API usage**: Uses Stripe's balance transactions API for minimal API calls
- **Standardized Naming**: Files named with date, amount, currency and payout ID

## Installation

1. Clone this repository
2. Install dependencies:

```bash
cd collect_my_payout_docs
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure your environment:

```bash
cp .env.example .env
# Edit .env with your Stripe API key
```

## Usage

The command requires year and month parameters:

```bash
python3 collect_my_payout_docs.py YEAR MONTH
```

For example, to fetch all payout documents for May 2023:

```bash
python3 collect_my_payout_docs.py 2023 5
```

### Example Output

```
payouts: 100%|████████████████████| 3/3 [00:09<00:00, 3.16s/it]
✔ saved 2023-05-15_1582.33eur_po_1NlXYZ.pdf to ~/Downloads
✔ saved 2023-05-22_-314.64eur_po_1NlABC.pdf to ~/Downloads
✔ saved 2023-05-30_764.90eur_po_1NlDEF.pdf to ~/Downloads
```

## File Format

Files are saved directly to your ~/Downloads folder with this naming pattern:

```
YYYY-MM-DD_amount_currency_payoutid.pdf
```

Where:
- **Date**: UTC arrival date of the payout
- **Amount**: Net amount of the payout (negative for withdrawals)
- **Currency**: Lowercase ISO code (eur, usd, etc.)
- **Payout ID**: The Stripe payout ID that appears in your bank reference

## Requirements

- Python 3.9+
- Stripe API key (with read-only access)
- Required packages:
  - stripe
  - pypdf
  - requests
  - python-dateutil
  - tqdm

## How It Works

1. Lists all payouts for the specified month
2. For each payout, fetches all balance transactions
3. Extracts PDF URLs for all invoices and credit notes
4. Downloads and merges all PDFs into a single file using PdfWriter
5. Saves with a standardized filename directly to ~/Downloads

## Security Notes

- Uses read-only Stripe API access
- Handles potentially sensitive customer information - store files securely
- No data is sent to external services beyond Stripe API
