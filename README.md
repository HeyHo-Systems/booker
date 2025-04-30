# Booker

A bookkeeping toolkit with two complementary Python tools:

1. **rename_my_invoices** - Rename and organize invoice PDFs and images using AI
2. **match_my_statements** - Match credit card statements against renamed invoices

## Quick Links

- [Rename My Invoices Documentation](rename_my_invoices/README.md)
- [Match My Statements Documentation](match_my_statements/README.md)

## Overview

This project provides a complete workflow for organizing financial documents:

1. First, use `rename_my_invoices` to process your invoice PDFs and images:
   - Extract dates, amounts, and payment methods using AI
   - Automatically rename and organize files based on payment type
   - Create consistent filenames for easy lookup

2. Then, use `match_my_statements` to reconcile your statements:
   - Process credit card statements and find matching invoices
   - Generate detailed reports of matched and unmatched items
   - Identify missing documentation

## Project Structure

```
booker/
├── match_my_statements/       # Statement matching tool
│   ├── match_my_statements.py # Main script
│   ├── README.md              # Documentation
│   ├── requirements.txt       # Dependencies
│   └── .env.example           # Sample environment config
│
├── rename_my_invoices/        # Invoice renaming tool
│   ├── rename_my_invoices.py  # Main script
│   ├── README.md              # Documentation
│   ├── requirements.txt       # Dependencies
│   └── .env.example           # Sample environment config
│
└── README.md                  # This file
```

## Installation

Each tool can be installed independently:

```bash
# For the rename_my_invoices tool
cd rename_my_invoices
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Edit with your API key
```

```bash
# For the match_my_statements tool
cd match_my_statements
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Edit with your API key
```

## Dependencies

Each tool has its own specific dependencies listed in its respective requirements.txt file.

Both tools require:
- Python 3.9+
- OpenAI API access

## Usage

For detailed usage instructions, see the tool-specific documentation linked above.

## License

MIT