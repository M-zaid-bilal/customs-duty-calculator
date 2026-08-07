import sys
import os

# Append Replit local packages path if needed
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

from pypdf import PdfReader
import chromadb

DATA_DIR = "data"
VECTOR_DIR = "vector_store"

# All 5 legal acts & rules for RAG
LEGAL_ACTS = [
    "Customs_Act_1969.pdf",
    "Sales_Tax_Act_1990.pdf",
    "FED_Act_2005.pdf",
    "Income_Tax_Ordinance_2001.pdf",
    "Custom_Rules_2001.pdf",
]


def extract_chunks_from_pdf(file_path, chunk_size=600, chunk_overlap=80):
    """
    Extracts text from a PDF file and splits it into overlapping text chunks.
    """
    reader = PdfReader(file_path)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {i + 1}]\n" + text

    words = full_text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def build_vector_database():
    print("==================================================")
    print("LAYER 3: VECTOR DATABASE CREATION (CHROMADB)")
    print("==================================================")

    # Initialize Persistent ChromaDB Client
    client = chromadb.PersistentClient(path=VECTOR_DIR)

    # Reset collection if running fresh
    try:
        client.delete_collection("customs_legal_acts")
    except Exception:
        pass

    collection = client.create_collection(
        name="customs_legal_acts",
        metadata={
            "description": "Vector store for Pakistan Customs Legal Acts & Rules"
        },
    )

    total_chunks = 0

    for pdf_filename in LEGAL_ACTS:
        file_path = os.path.join(DATA_DIR, pdf_filename)

        if not os.path.exists(file_path):
            print(
                f"⚠️ Warning: File '{pdf_filename}' not found in {DATA_DIR}/. Skipping."
            )
            continue

        print(f"📖 Processing: {pdf_filename} ...")
        chunks = extract_chunks_from_pdf(file_path)

        if not chunks:
            print(f"⚠️ Warning: No readable text extracted from {pdf_filename}.")
            continue

        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{pdf_filename}_chunk_{idx}"
            documents.append(chunk)
            metadatas.append({"source": pdf_filename, "chunk_index": idx})
            ids.append(chunk_id)

        # Batch add to ChromaDB
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        total_chunks += len(documents)
        print(f"  └─ Inserted {len(documents)} text chunks.")

    print("\n==================================================")
    print(f"✅ Vector database successfully built with {total_chunks} legal chunks!")
    print(f"📁 Stored at: '{VECTOR_DIR}/'")
    print("==================================================")


def query_test(query_text="penalty for non-payment of duty"):
    print("\n==================================================")
    print(f"🔍 TESTING RETRIEVAL QUERY: '{query_text}'")
    print("==================================================")

    client = chromadb.PersistentClient(path=VECTOR_DIR)
    collection = client.get_collection(name="customs_legal_acts")

    results = collection.query(query_texts=[query_text], n_results=2)

    # Safe checking to prevent NoneType subscript errors
    docs = results.get("documents", [[]])[0] if results.get("documents") else []  # ty:ignore[not-subscriptable]
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []  # ty:ignore[not-subscriptable]

    if not docs:
        print("⚠️ No matching legal provisions found.")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("source", "Unknown") if meta else "Unknown"
        print(f"\n--- Match {i + 1} [Source: {source}] ---")
        print(f"{doc[:300]}...\n")


if __name__ == "__main__":
    build_vector_database()
    query_test()
