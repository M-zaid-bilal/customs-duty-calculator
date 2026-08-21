import sys
import os
import re
import math
import json
import sqlite3
from typing import List, Dict, Any, cast

# Append local packages path if needed
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

from pypdf import PdfReader

DATA_DIR = "data"
VECTOR_DIR = "vector_store"
VECTOR_DB_PATH = os.path.join(VECTOR_DIR, "legal_chunks.sqlite3")
VECTOR_DIM = 128

# Default baseline fallback list of legal acts
DEFAULT_LEGAL_ACTS = [
    "Customs_Act_1969.pdf",
    "Sales_Tax_Act_1990.pdf",
    "FED_Act_2005.pdf",
    "Income_Tax_Ordinance_2001.pdf",
    "Custom_Rules_2001.pdf",
]


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words longer than 2 characters."""
    return [w.lower() for w in re.findall(r"\w+", text or "") if len(w) > 2]


def _embed_text(text: str) -> List[float]:
    """
    Pure-Python term-frequency embedding function using deterministic hashing.
    Bypasses external C++ DLL and ONNX runtime dependencies.
    """
    tokens = _tokenize(text)
    vec = [0.0] * VECTOR_DIM
    if not tokens:
        return vec

    for token in tokens:
        idx = abs(hash(token)) % VECTOR_DIM
        vec[idx] += 1.0

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two numeric vectors."""
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return float(dot / (mag1 * mag2))


def _ensure_db() -> None:
    """Ensures vector_store directory exists and initializes SQLite table schema."""
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with sqlite3.connect(VECTOR_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legal_chunks (
                id TEXT PRIMARY KEY,
                source TEXT,
                text TEXT,
                embedding TEXT
            )
            """
        )


def discover_pdf_files(data_dir: str) -> List[str]:
    """
    Discovers all PDF files present in the data directory.
    Automatically includes any newly added SROs, classification rulings, or schedules.
    """
    if not os.path.exists(data_dir):
        return []

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    return sorted(pdf_files)


def extract_chunks_from_pdf(file_path: str, chunk_size: int = 600, chunk_overlap: int = 80) -> List[str]:
    """Extracts text from a PDF file and splits it into overlapping word chunks."""
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"⚠️ Error reading PDF file '{file_path}': {e}")
        return []

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
    """Reads all PDF documents in data/, extracts text chunks, embeds them, and stores in SQLite."""
    print("==================================================")
    print("LAYER 3: VECTOR DATABASE CREATION (SQLITE VECTOR STORE)")
    print("==================================================")

    _ensure_db()

    pdf_files = discover_pdf_files(DATA_DIR)
    if not pdf_files:
        pdf_files = DEFAULT_LEGAL_ACTS

    print(f"📁 Discovered {len(pdf_files)} PDF document(s) in '{DATA_DIR}/': {pdf_files}\n")

    with sqlite3.connect(VECTOR_DB_PATH) as conn:
        # Clear previous indices for a clean build
        conn.execute("DELETE FROM legal_chunks")

        total_chunks = 0
        for pdf_filename in pdf_files:
            file_path = os.path.join(DATA_DIR, pdf_filename)
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: File '{pdf_filename}' not found in {DATA_DIR}/. Skipping.")
                continue

            print(f"📖 Processing: {pdf_filename} ...")
            chunks = extract_chunks_from_pdf(file_path)
            if not chunks:
                print(f"⚠️ Warning: No readable text extracted from {pdf_filename}.")
                continue

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{pdf_filename}_chunk_{idx}"
                embedding_json = json.dumps(_embed_text(chunk))
                conn.execute(
                    "INSERT INTO legal_chunks (id, source, text, embedding) VALUES (?, ?, ?, ?)",
                    (chunk_id, pdf_filename, chunk, embedding_json),
                )
                total_chunks += 1

            print(f"  └─ Inserted {len(chunks)} text chunks.")

    print("\n==================================================")
    print(f"✅ Vector database successfully built with {total_chunks} legal chunks!")
    print(f"📁 Stored at: '{VECTOR_DB_PATH}'")
    print("==================================================")


def query_legal_chunks(query_text: str, n_results: int = 2) -> List[Dict[str, str]]:
    """Retrieves top N matching legal chunks from SQLite vector store based on cosine similarity."""
    _ensure_db()
    query_vector = _embed_text(query_text)

    with sqlite3.connect(VECTOR_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT source, text, embedding FROM legal_chunks"
        ).fetchall()

    if not rows:
        return []

    scored: List[tuple[float, str, str]] = []
    for source, text, embedding_json in rows:
        embedding = json.loads(embedding_json) if embedding_json else [0.0] * VECTOR_DIM
        score = _cosine_similarity(query_vector, embedding)
        scored.append((score, source, text))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"source": source, "text": text.strip()}
        for _, source, text in scored[:n_results]
    ]


def query_test(query_text="penalty for non-payment of duty"):
    """Demonstrates vector retrieval search on built SQLite database."""
    print("\n==================================================")
    print(f"🔍 TESTING RETRIEVAL QUERY: '{query_text}'")
    print("==================================================")

    results = query_legal_chunks(query_text, n_results=2)
    if not results:
        print("⚠️ No matching legal provisions found.")
        return

    for i, result in enumerate(results, 1):
        source = result.get("source", "Unknown")
        text = result.get("text", "")
        print(f"\n--- Match {i} [Source: {source}] ---")
        print(f"{text[:300]}...\n")


if __name__ == "__main__":
    build_vector_database()
    query_test()