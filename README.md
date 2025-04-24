# Rename-my-Invoices (v0.1)

A single-file Python utility that scans a folder of PDFs / image receipts, asks an OpenAI model for four bookkeeping fields, and renames each file to `YYYY.MM.DD_method_amount_usage.ext`.

## 1. Install

Create a fresh environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install openai pdfminer.six pytesseract pillow rich
```

System dependencies:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`

## 2. Configure

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-…"
```
You can also point the `OPENAI_API_BASE` / `OPENAI_API_KEY` variables to your compatible proxy.

Note: Inside `rename_agent.py` the model defaults to `gpt-4o-mini`; change `model=` if you prefer another chat model or a local endpoint.

## 3. Run

Try a dry run first:
```bash
python rename_agent.py ~/invoices --dry-run
```

If the output looks good, commit:
```bash
python rename_agent.py ~/invoices
```

### Options

| Flag | Effect |
|------|--------|
| `--dry-run` | Print the proposed new names but don't rename anything |
| `--log rename_log.jsonl` | Append one JSON line per file (old, new, fields) for traceability |

Note: The script ignores hidden files, sub-directories, and non-PDF/image extensions.

## 4. How it works (quick tour)

1. **Extract text**
   - PDF → `pdfminer.six.extract_text()`
   - Image → `pytesseract.image_to_string()`

2. **LLM call**
   - Sends the first 3,500 chars to Chat Completions with a function schema:
   ```json
   {
     "date": "YYYY-MM-DD",
     "method": "visa",
     "amount": "123.45 EUR",
     "usage": "fuel"
   }
   ```

3. **Validation & normalisation**
   - Basic regex checks
   - Normalises amounts (123.45EUR → 123.45 EUR)
   - Usage to snake_case ≤ 30 chars

4. **Rename**
   - Builds `YYYY.MM.DD_method_amount_usage.ext` (dots in date, original extension preserved)
   - Calls `Path.rename()`

## 5. Next steps

- Watch mode – add watchdog to trigger on new files
- Better OCR – swap in Unstructured-IO or PaddleOCR for scans
- Bulk / async – wrap the extraction in `asyncio.gather()` or a LangChain Runnable for speed
- Undo UI – small Streamlit page that loads the JSONL log and lets you revert

Happy hacking!