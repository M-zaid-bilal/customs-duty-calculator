# Customs Duty Calculator (Pakistan)

An AI-powered engine that calculates the full Pakistan import duty & tax liability for a shipment — Customs Duty, Federal Excise Duty, Sales Tax, Advance Income Tax, and Sindh Infrastructure Cess — and grounds the result in retrieved excerpts from Pakistan's actual customs legislation.

Give it an HS Code (or just a plain-English item description), an FOB value, and freight/insurance, and it resolves the correct tariff line, runs the statutory duty waterfall, retrieves relevant legal context via RAG, and synthesizes a readable assessment report with Gemini.

## How it works

The pipeline is organized into layers:

| Layer | File | Responsibility |
|---|---|---|
| 1 — Ingestion | `scripts/ingest_and_inspect.py` | Parses the FBR tariff schedule PDF into a clean SQLite table (`pct_tariff`), and seeds Sindh Infrastructure Cess slabs |
| 2 — Exact Lookup | `scripts/calculator.py` | SQLite lookups for HS-code duty rate and cess slab, given an assessed value |
| 3 — Legal RAG | `scripts/build_vector_store.py` | Chunks 5 core legal PDFs (Customs Act, Sales Tax Act, FED Act, Income Tax Ordinance, Customs Rules) and embeds them into a persistent ChromaDB collection |
| 4 — Math Engine | `scripts/calculator.py` | Runs the sequential compounding duty waterfall (CD → FED → Sales Tax → AIT → Sindh Cess) |
| 5 — Orchestrator | `scripts/orchestrator.py` | Resolves the HS code, runs the math engine, retrieves legal context, and calls Gemini to produce a final markdown assessment — with exact-hash and semantic (cosine-similarity) caching on top |

### Duty waterfall

Each tax is computed on a progressively larger base, matching how Pakistan Customs actually stacks these charges:

1. **CIF** = FOB + Freight + Insurance, converted to PKR
2. **Assessed Value (AV)** = CIF × 1.01 (1% landing charges)
3. **Customs Duty (CD)** = AV × CD rate (looked up by HS code)
4. **Federal Excise Duty (FED)** = (AV + CD) × FED rate
5. **Sales Tax (ST)** = (AV + CD + FED) × 18% (default)
6. **Advance Income Tax (AIT)** = (AV + CD + FED + ST) × AIT rate (6% for filers, higher for non-filers)
7. **Sindh Infrastructure Cess** = AV × slab rate (0.0180 – 0.0185, based on assessed value bracket)
8. **Total Payable** = CD + FED + ST + AIT + Sindh Cess

### HS code resolution

`resolve_hs_code()` in `orchestrator.py` handles three cases:
- **Full 8-digit code** → exact match against `pct_tariff`
- **Partial prefix** (e.g. `8703`) → expands to matching 8-digit leaf codes, then picks the best match by scoring keyword overlap against the item description
- **No code at all** → searches tariff descriptions for the item description directly

### Caching

The orchestrator uses a 3-tier cache to avoid redundant LLM calls:
1. **Exact hash cache** — identical query + params → instant return
2. **Semantic cache** — cosine similarity ≥ 0.88 on tokenized query text, with identical numeric params → reused report
3. **Gemini context cache** — the system prompt itself is cached server-side (1 hour TTL) to cut token cost across calls

## Repo structure

```
.
├── data/                   # Source PDFs: FBR tariff schedule + legal acts/rules
├── db/
│   └── customs_master.db   # SQLite: pct_tariff (8,300+ HS codes) + sindh_cess slabs
├── vector_store/           # Persistent ChromaDB collection of embedded legal text
├── scripts/
│   ├── ingest_and_inspect.py   # Layer 1: PDF → SQLite ingestion
│   ├── build_vector_store.py   # Layer 3: PDF → ChromaDB embeddings
│   ├── calculator.py           # Layer 2 & 4: lookups + duty waterfall math
│   ├── exchange_rates.py       # Live USD/PKR rate fetch, cached, with fallback
│   ├── orchestrator.py         # Layer 5: full pipeline + caching + Gemini synthesis
│   └── test_layers.py          # Manual smoke tests for Layers 2 & 3
└── tests/
    ├── test_calculator.py      # Unit tests for the duty math engine
    └── test_pipeline.py        # Integration tests across all layers (mocked I/O)
```

## Setup

```bash
git clone https://github.com/M-zaid-bilal/customs-duty-calculator.git
cd customs-duty-calculator
pip install -r requirements.txt
```

Set your Gemini API key (used for report synthesis):

```bash
export GEMINI_API_KEY=your_key_here   # or GOOGLE_API_KEY
```

Optionally pin the exchange rate instead of hitting the live API:

```bash
export USD_PKR_RATE=280.0
```

The repo ships with `db/customs_master.db` and `vector_store/` already built. To rebuild them from the source PDFs in `data/`:

```bash
python scripts/ingest_and_inspect.py     # rebuilds pct_tariff + sindh_cess in SQLite
python scripts/build_vector_store.py     # rebuilds the ChromaDB legal embeddings
```

## Usage

```python
from scripts.orchestrator import run_customs_orchestrator

result = run_customs_orchestrator(
    user_query="I want to import a mini van",
    hs_code="8703",              # full, partial, or blank
    item_description="Mini Van",
    fob_usd=15000,
    freight_usd=500,
    insurance_usd=100,
    ait_rate=6.0,                 # 6% filer / 12% non-filer
    fed_rate=0.0,
)

print(result["ai_report"])        # Gemini-generated markdown assessment
print(result["calculation"])      # Full numeric duty breakdown
```

Or run the math engine directly, without the LLM/RAG layers:

```python
from scripts.calculator import calculate_customs_duty

calculate_customs_duty(fob_usd=10000, hs_code="87032113", ait_rate=6.0)
```

Run the demo scripts directly for a walkthrough of each layer:

```bash
python scripts/calculator.py       # duty math engine demo
python scripts/exchange_rates.py   # exchange rate caching demo
python scripts/test_layers.py      # SQLite + ChromaDB lookup demo
python scripts/orchestrator.py     # full pipeline + caching demo
```

## Testing

```bash
python -m unittest tests/test_calculator.py
pytest tests/test_pipeline.py
```

## Notes

- The tariff database (`pct_tariff`) currently holds 8,300+ 8-digit HS code entries parsed from the 2025–2026 FBR tariff schedule.
- Sindh Infrastructure Cess is applied based on assessed value slabs; rates and brackets are hardcoded as a fallback if the `sindh_cess` table is unavailable.
- Legal RAG context is drawn from five source acts: Customs Act 1969, Sales Tax Act 1990, FED Act 2005, Income Tax Ordinance 2001, and Customs Rules 2001.
- Built and run originally on Replit (Python 3.11).
- This tool provides an estimate for informational purposes only — final duty assessment is determined solely by Pakistan Customs / FBR.