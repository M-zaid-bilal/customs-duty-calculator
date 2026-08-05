import sys
import os

# Append Replit local packages path
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

import sqlite3
import pdfplumber  # ty:ignore[unresolved-import]

DB_PATH = "db/customs_master.db"
DATA_DIR = "data"


def inspect_raw_files():
    print("==================================================")
    print("1. RAW FILE INSPECTION")
    print("==================================================")
    if not os.path.exists(DATA_DIR):
        print(f"❌ Directory '{DATA_DIR}' does not exist.")
        return False

    files = os.listdir(DATA_DIR)
    pdf_files = [f for f in files if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDF files found in '{DATA_DIR}/'. Please upload your PDFs.")
        return False

    print(f"Found {len(pdf_files)} PDF files in '{DATA_DIR}/':")
    for f in pdf_files:
        size_mb = os.path.getsize(os.path.join(DATA_DIR, f)) / (1024 * 1024)
        print(f"  • {f} ({size_mb:.2f} MB)")
    print()
    return True


def setup_database():
    print("==================================================")
    print("2. SQLITE DATABASE INGESTION")
    print("==================================================")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Create Tariff Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pct_tariff (
            hs_code TEXT PRIMARY KEY,
            description TEXT,
            cd_rate REAL
        )
    """)
    # Clear previous table entries for a clean re-ingestion
    cursor.execute("DELETE FROM pct_tariff")

    # 2. Create & Populate Sindh Infrastructure Cess Slabs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sindh_cess (
            min_weight REAL,
            max_weight REAL,
            cess_rate REAL,
            flat_km_rate REAL
        )
    """)

    cursor.execute("DELETE FROM sindh_cess")
    sindh_slabs = [
        (0.0, 1250.0, 0.0180, 0.01),
        (1250.0, 2030.0, 0.0181, 0.01),
        (2030.0, 4060.0, 0.0182, 0.01),
        (4060.0, 8120.0, 0.0183, 0.01),
        (8120.0, 16000.0, 0.0184, 0.01),
        (16000.0, 999999.0, 0.0185, 0.01),
    ]
    cursor.executemany("INSERT INTO sindh_cess VALUES (?,?,?,?)", sindh_slabs)
    print("✅ Sindh Infrastructure Cess slabs initialized.")

    # 3. Ingest Pakistan Customs Tariff
    tariff_pdf_path = None
    for f in os.listdir(DATA_DIR):
        if "tarrif" in f.lower() or "tariff" in f.lower():
            if f.endswith(".pdf"):
                tariff_pdf_path = os.path.join(DATA_DIR, f)
                break

    if tariff_pdf_path:
        print(f"📖 Parsing tariff from: {tariff_pdf_path}")
        records_inserted = 0
        skipped_headers = 0

        with pdfplumber.open(tariff_pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    for row in table[1:]:
                        if len(row) >= 3 and row[0]:
                            raw_code = row[0].strip().replace(".", "")

                            # Filter out table headers and non-digit HS entries
                            if not raw_code or not raw_code[0].isdigit():
                                skipped_headers += 1
                                continue

                            description = row[1].strip() if row[1] else ""
                            try:
                                cd_rate = float(row[2].strip().replace("%", "")) / 100.0
                            except (ValueError, AttributeError):
                                cd_rate = 0.0

                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO pct_tariff (hs_code, description, cd_rate)
                                VALUES (?, ?, ?)
                            """,
                                (raw_code, description, cd_rate),
                            )
                            records_inserted += 1

        print(f"✅ Ingested {records_inserted} clean PCT Tariff records into SQLite.")
        print(f"🧹 Filtered out {skipped_headers} header/junk rows.")
    else:
        print("⚠️ Tariff PDF not found in data folder. Skipping tariff ingestion.")

    conn.commit()
    conn.close()


def inspect_database():
    print("\n==================================================")
    print("3. DATABASE INSPECTION REPORT")
    print("==================================================")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pct_tariff")
    pct_count = cursor.fetchone()[0]
    print(f"📊 Total Clean Records in 'pct_tariff': {pct_count}")

    if pct_count > 0:
        print("\n🔍 Sample Tariff Records (Headers Filtered):")
        cursor.execute("SELECT hs_code, description, cd_rate FROM pct_tariff LIMIT 5")
        for row in cursor.fetchall():
            print(f"  • HS: {row[0]} | Duty: {row[2] * 100}% | Desc: {row[1][:40]}...")

    cursor.execute("SELECT COUNT(*) FROM sindh_cess")
    cess_count = cursor.fetchone()[0]
    print(f"\n📊 Total Slabs in 'sindh_cess': {cess_count}")

    conn.close()


if __name__ == "__main__":
    if inspect_raw_files():
        setup_database()
        inspect_database()
