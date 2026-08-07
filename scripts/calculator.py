import sys
import os
import json
import sqlite3

# Ensure local directory & packages are accessible
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

# Import exchange rate engine from our exchange_rates module
from scripts.exchange_rates import get_usd_to_pkr_rate as fetch_usd_to_pkr

DB_PATH = "db/customs_master.db"


# ==========================================
# 1. TARIFF & CESS LOOKUPS FROM SQLITE
# ==========================================
def get_customs_duty_rate(hs_code: str) -> float:
    """Queries SQLite database to fetch exact Custom Duty (CD) percentage."""
    if not os.path.exists(DB_PATH):
        return 0.0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT cd_rate FROM pct_tariff WHERE hs_code = ?", (hs_code,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0] is not None:
        return float(row[0])
    return 0.0


def get_sindh_cess_rate(assessed_value_pkr: float) -> float:
    """Queries SQLite database or applies standard Sindh Cess slab brackets."""
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT rate_percent FROM sindh_cess 
                WHERE ? >= min_value_pkr AND ? <= max_value_pkr
                LIMIT 1
            """,
                (assessed_value_pkr, assessed_value_pkr),
            )

            row = cursor.fetchone()
            conn.close()
            if row and row[0] is not None:
                return float(row[0])
        except Exception:
            pass

    # Standard Fallback Slabs
    if assessed_value_pkr <= 1250000:
        return 1.80
    elif assessed_value_pkr <= 2030000:
        return 1.81
    elif assessed_value_pkr <= 4060000:
        return 1.82
    elif assessed_value_pkr <= 8120000:
        return 1.83
    elif assessed_value_pkr <= 16000000:
        return 1.84
    else:
        return 1.85


# ==========================================
# 2. LAYER 4 CORE MATH ENGINE (WATERFALL)
# ==========================================
def calculate_customs_duty(
    fob_usd: float,
    hs_code: str,
    ait_rate: float = 6.0,  # Advance Income Tax %
    sales_tax_rate: float = 18.0,  # Standard Sales Tax %
    fed_rate: float = 0.0,  # Federal Excise Duty %
    freight_usd: float = 0.0,
    insurance_usd: float = 0.0,
    exchange_rate: float
    | None = None,  # FIXED: Default set to None so API gets called!
) -> dict:
    """
    Executes Pakistan Customs sequential compounding duty calculation waterfall.
    """
    # 1. Fetch Exchange Rate from exchange_rates.py if not manually provided
    if exchange_rate is None or exchange_rate <= 0:
        exchange_rate = fetch_usd_to_pkr()

    cd_rate = get_customs_duty_rate(hs_code)

    # 2. CIF & Assessed Value (AV)
    cif_usd = fob_usd + freight_usd + insurance_usd
    cif_pkr = cif_usd * exchange_rate
    landing_charges_pkr = cif_pkr * 0.01  # Standard 1% Landing Charges
    assessed_value_pkr = cif_pkr + landing_charges_pkr

    # 3. Customs Duty (CD)
    cd_amount = assessed_value_pkr * (cd_rate / 100.0)

    # 4. Federal Excise Duty (FED) -> Base: AV + CD
    fed_base = assessed_value_pkr + cd_amount
    fed_amount = fed_base * (fed_rate / 100.0)

    # 5. Sales Tax (ST) -> Base: AV + CD + FED
    sales_tax_base = assessed_value_pkr + cd_amount + fed_amount
    sales_tax_amount = sales_tax_base * (sales_tax_rate / 100.0)

    # 6. Advance Income Tax (AIT) -> Base: AV + CD + FED + ST
    ait_base = assessed_value_pkr + cd_amount + fed_amount + sales_tax_amount
    ait_amount = ait_base * (ait_rate / 100.0)

    # 7. Sindh Infrastructure Cess -> Base: AV
    sindh_cess_rate = get_sindh_cess_rate(assessed_value_pkr)
    sindh_cess_amount = assessed_value_pkr * (sindh_cess_rate / 100.0)

    # 8. Total Duty & Tax Liability
    total_duty_payable = (
        cd_amount + fed_amount + sales_tax_amount + ait_amount + sindh_cess_amount
    )

    return {
        "inputs": {
            "hs_code": hs_code,
            "fob_usd": fob_usd,
            "exchange_rate": exchange_rate,
            "cif_pkr": round(cif_pkr, 2),
            "assessed_value_pkr": round(assessed_value_pkr, 2),
        },
        "rates_applied": {
            "cd_rate": cd_rate,
            "fed_rate": fed_rate,
            "sales_tax_rate": sales_tax_rate,
            "ait_rate": ait_rate,
            "sindh_cess_rate": sindh_cess_rate,
        },
        "duties_breakdown_pkr": {
            "customs_duty": round(cd_amount, 2),
            "federal_excise_duty": round(fed_amount, 2),
            "sales_tax": round(sales_tax_amount, 2),
            "advance_income_tax": round(ait_amount, 2),
            "sindh_cess": round(sindh_cess_amount, 2),
            "total_payable": round(total_duty_payable, 2),
        },
    }


# ==========================================
# TEST RUN
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("LAYER 4: DUTY CALCULATION ENGINE TEST")
    print("==================================================")

    result = calculate_customs_duty(
        fob_usd=10000, hs_code="87032113", ait_rate=6.0, fed_rate=0.0
    )

    print(json.dumps(result, indent=2))
