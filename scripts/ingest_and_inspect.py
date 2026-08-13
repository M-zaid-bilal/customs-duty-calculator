import sys
import os
import cffi

# Append Replit local packages path
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

import sqlite3
import pdfplumber

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

    # 2b. Create HS Code Transposition Table (HS-2017 <-> HS-2022)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hs_transposition (
            old_hs_code TEXT,
            old_description TEXT,
            new_hs_code TEXT,
            new_description TEXT,
            remarks TEXT
        )
    """)
    cursor.execute("DELETE FROM hs_transposition")

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
                                cd_rate = float((row[2] or "").strip().replace("%", "")) / 100.0
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

    # 4. Ingest HS Code Transposition Table (HS-2017 <-> HS-2022)
    transposition_pdf_path = None
    for f in os.listdir(DATA_DIR):
        if "transposition" in f.lower() and f.endswith(".pdf"):
            transposition_pdf_path = os.path.join(DATA_DIR, f)
            break

    if transposition_pdf_path:
        print(f"📖 Parsing HS transposition table from: {transposition_pdf_path}")
        transposition_rows = 0

        with pdfplumber.open(transposition_pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                for row in table:
                    # Expected columns: [Sr, HS-2017 Code, Desc, CD%, UoM, WTO, TypeOfChange,
                    #                     HS-2022Ref, HS-2022 Code, Desc, CD%, UoM, WTO, TypeOfChange,
                    #                     HS-2017Ref, Remarks]
                    if len(row) < 10:
                        continue

                    old_code = (row[1] or "").strip().replace(".", "")
                    old_desc = (row[2] or "").strip()
                    new_code = (row[8] or "").strip().replace(".", "")
                    new_desc = (row[9] or "").strip()
                    remarks = (row[15] or "").strip() if len(row) > 15 else ""

                    # Skip repeated header rows and pure heading/section rows (no PCT code at all)
                    if not old_code and not new_code:
                        continue
                    if old_code in ("PCT CODE", "HS-2017") or new_code in (
                        "PCT CODE",
                        "HS-2022",
                    ):
                        continue
                    if not (old_code[:1].isdigit() or new_code[:1].isdigit()):
                        continue

                    cursor.execute(
                        """
                        INSERT INTO hs_transposition
                        (old_hs_code, old_description, new_hs_code, new_description, remarks)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (old_code, old_desc, new_code, new_desc, remarks),
                    )
                    transposition_rows += 1

        print(f"✅ Ingested {transposition_rows} HS transposition records into SQLite.")
    else:
        print(
            "⚠️ HS Transposition PDF not found in data folder. Skipping transposition ingestion."
        )

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

    cursor.execute("SELECT COUNT(*) FROM hs_transposition")
    transposition_count = cursor.fetchone()[0]
    print(f"📊 Total Records in 'hs_transposition': {transposition_count}")

    conn.close()


if __name__ == "__main__":
    if inspect_raw_files():
        setup_database()
        inspect_database()