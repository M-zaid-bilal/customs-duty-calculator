import sys
import os
import sqlite3
import chromadb

# Ensure local packages are reachable
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

DB_PATH = "db/customs_master.db"
VECTOR_DIR = "vector_store"


# ==========================================
# LAYER 2: SQLITE EXACT LOOKUP TEST
# ==========================================
def query_sqlite(hs_code_query):
    print(f"\n==================================================")
    print(f"📊 LAYER 2 (SQLITE): Searching HS Code: '{hs_code_query}'")
    print(f"==================================================")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Search exact match or prefix match in pct_tariff
    cursor.execute(
        """
        SELECT hs_code, cd_rate, description 
        FROM pct_tariff 
        WHERE hs_code = ? OR hs_code LIKE ?
        LIMIT 5
    """,
        (hs_code_query, f"{hs_code_query}%"),
    )

    records = cursor.fetchall()

    if records:
        print(f"Found {len(records)} matching tariff record(s):")
        for hs, cd, desc in records:
            print(f"  • HS Code: {hs} | Custom Duty (CD): {cd}% | Description: {desc}")
    else:
        print(f"⚠️ No matching tariff record found for HS Code '{hs_code_query}'.")

    # Display Sindh Infrastructure Cess Slabs safely
    try:
        cursor.execute("SELECT * FROM sindh_cess")
        cess_slabs = cursor.fetchall()
        print("\n  [Sindh Cess Rate Slabs Reference]:")
        for row in cess_slabs:
            print(f"    - {row}")
    except Exception as e:
        print(f"⚠️ Could not fetch sindh_cess table: {e}")

    conn.close()


# ==========================================
# LAYER 3: CHROMADB SEMANTIC LEGAL SEARCH
# ==========================================
def query_vector_store(legal_query, n_results=2):
    print(f"\n==================================================")
    print(f"🔍 LAYER 3 (CHROMADB): Legal Query: '{legal_query}'")
    print(f"==================================================")

    client = chromadb.PersistentClient(path=VECTOR_DIR)
    collection = client.get_collection(name="customs_legal_acts")

    results = collection.query(query_texts=[legal_query], n_results=n_results)

    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    if not docs:
        print("⚠️ No matching legal provisions found.")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("source", "Unknown") if meta else "Unknown"
        # Print neat snippet
        print(f"\n--- Match {i + 1} [Source: {source}] ---")
        cleaned_doc = doc.replace("\n", " ").strip()
        print(f"{cleaned_doc[:350]}...\n")


# ==========================================
# RUN TEST CASES
# ==========================================
if __name__ == "__main__":
    # --- TEST CASE 1: Horses & Live Animals ---
    query_sqlite(hs_code_query="0101")
    query_vector_store(legal_query="exemption or restriction on import of live animals")

    print("\n" + "#" * 60 + "\n")

    # --- TEST CASE 2: Automobiles & Vehicles ---
    query_sqlite(hs_code_query="8703")
    query_vector_store(
        legal_query="confiscation and valuation of imported motor vehicles"
    )
