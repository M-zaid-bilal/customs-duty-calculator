import sys
import os
import json
import sqlite3
import re
import hashlib
import time
import math
from collections import Counter
from typing import Optional, Dict, Any, List, cast

# Ensure local packages & paths are accessible
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

from google import genai
from google.genai import types

# Import Layer 4 Math Engine
from scripts.calculator import calculate_customs_duty, DB_PATH
from scripts.build_vector_store import query_legal_chunks

VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "customs_legal_acts"

# In-Memory Cache Containers
_EXACT_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_SEMANTIC_CACHE: List[
    Dict[str, Any]
] = []  # Stores {"query_vec": Counter, "params": dict, "result": dict}

# Global Gemini Context Cache handle
_GEMINI_CACHE_HANDLE: Optional[str] = None
_GEMINI_CACHE_EXPIRY: float = 0.0


# ==========================================
# SIMILARITY HELPERS FOR SEMANTIC CACHING
# ==========================================
def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words (>2 chars)."""
    return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 2]


def _vectorize(tokens: List[str]) -> Counter:
    """Converts tokens into term-frequency counter."""
    return Counter(tokens)


def _cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """Calculates Cosine Similarity between two term-frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    magnitude = math.sqrt(sum1) * math.sqrt(sum2)

    if not magnitude:
        return 0.0
    return float(dot_product) / magnitude


# ==========================================
# GEMINI CONTEXT CACHING HELPER
# ==========================================
def get_or_create_context_cache(
    client: genai.Client, model: str = "gemini-3.1-flash-lite"
) -> Optional[str]:
    """
    Creates or retrieves an explicit Gemini CachedContent resource.
    Caches system rules to reduce token consumption on repeated calls.
    """
    global _GEMINI_CACHE_HANDLE, _GEMINI_CACHE_EXPIRY
    current_time = time.time()

    if _GEMINI_CACHE_HANDLE and current_time < _GEMINI_CACHE_EXPIRY:
        return _GEMINI_CACHE_HANDLE

    try:
        base_instructions = """
You are an expert Pakistan Customs & Trade Consultant AI.
Analyze the provided duty calculation breakdown and legal context snippets to generate a concise, professional assessment report.

Formatting Guidelines:
- State the resolved HS code and official FBR tariff description.
- Present a clean, markdown-formatted duty breakdown table.
- Highlight applicable legal rules, exemptions, or valuation conditions retrieved from the legal context.
- Keep tone authoritative, clear, and scannable.
"""
        # Pass base_instructions directly as a string or system_instruction
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                contents=base_instructions.strip(),
                ttl="3600s",  # 1 Hour TTL
            ),
        )
        _GEMINI_CACHE_HANDLE = cache.name
        _GEMINI_CACHE_EXPIRY = current_time + 3500
        print(f"✅ Created Gemini Context Cache: {cache.name}")
        return _GEMINI_CACHE_HANDLE
    except Exception as e:
        print(
            f"ℹ️ Context Cache Notice: {e} (Falling back to standard prompt execution)"
        )
        return None


# ==========================================
# 1. HS CODE VALIDATOR & RESOLVER
# ==========================================
def resolve_transposed_hs_code(clean_hs: str) -> str:
    """
    Looks up 'clean_hs' as a legacy HS-2017 code in the hs_transposition table
    and returns its current HS-2022 equivalent if a mapping exists and differs.
    Falls back to returning the input unchanged if no mapping is found.
    """
    if not clean_hs:
        return clean_hs

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT new_hs_code FROM hs_transposition WHERE old_hs_code = ? LIMIT 1",
            (clean_hs,),
        )
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        # hs_transposition table not yet ingested
        row = None
    finally:
        conn.close()

    if row and row[0] and row[0] != clean_hs:
        print(f"🔄 HS code transposed: {clean_hs} (HS-2017) → {row[0]} (HS-2022)")
        return row[0]
    return clean_hs


def resolve_hs_code(hs_input: str, item_description: str = "") -> Dict[str, Any]:
    """Validates, searches, and dynamically resolves HS codes from user input."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    desc_clean = (item_description or "").strip()

    if not hs_input or hs_input.strip() == "":
        search_terms = [
            term.lower()
            for term in re.findall(r"\w+", desc_clean)
            if len(term) > 2 and term.lower() not in {"want", "import", "into", "for", "the", "a", "an", "of", "and", "with", "i", "please", "goods", "product", "item", "vehicle", "vehicles"}
        ]

        matches = []
        if search_terms:
            seen = set()
            for term in search_terms:
                cursor.execute(
                    "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE LOWER(description) LIKE ? AND LENGTH(hs_code) = 8 ORDER BY hs_code LIMIT 20",
                    (f"%{term}%",),
                )
                for row in cursor.fetchall():
                    key = (row[0], row[1])
                    if key not in seen:
                        seen.add(key)
                        matches.append(row)

        if not matches:
            query_term = f"%{desc_clean.lower()}%"
            cursor.execute(
                "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE LOWER(description) LIKE ? AND LENGTH(hs_code) = 8 LIMIT 5",
                (query_term,),
            )
            matches = cursor.fetchall()

        conn.close()

        if matches:
            matched_rows = matches[:5]
            best_match = matched_rows[0]
            return {
                "status": "AUTO_RESOLVED",
                "selected_hs_code": best_match[0],
                "description": best_match[1],
                "cd_rate": best_match[2],
                "alternatives": [
                    {"hs_code": m[0], "description": m[1]} for m in matched_rows[1:]
                ],
            }
        else:
            return {
                "status": "NOT_FOUND",
                "message": f"Could not auto-detect HS code for '{item_description}'. Please specify an 8-digit HS Code.",
            }

    clean_hs = re.sub(r"\D", "", hs_input)
    original_clean_hs = clean_hs
    clean_hs = resolve_transposed_hs_code(clean_hs)
    transposed_from = original_clean_hs if clean_hs != original_clean_hs else None

    if len(clean_hs) < 8:
        prefix = clean_hs + "%"
        cursor.execute(
            "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code LIKE ? AND LENGTH(hs_code) = 8",
            (prefix,),
        )
        leaf_nodes = cursor.fetchall()
        conn.close()

        if not leaf_nodes:
            return {
                "status": "INVALID_PREFIX",
                "message": f"No tariff records found starting with prefix '{clean_hs}'.",
            }

        best_match = None
        if desc_clean:
            keywords = [
                kw.lower() for kw in re.findall(r"\w+", desc_clean) if len(kw) > 2
            ]
            highest_score = 0
            for node in leaf_nodes:
                node_hs, node_desc, node_cd = node
                node_desc_lower = node_desc.lower()
                score = sum(1 for kw in keywords if kw in node_desc_lower)
                if score > highest_score:
                    highest_score = score
                    best_match = node

        selected_node = best_match if best_match else leaf_nodes[0]

        return {
            "status": "PARTIAL_CODE_EXPANDED",
            "heading_code": clean_hs,
            "selected_hs_code": selected_node[0],
            "description": selected_node[1],
            "cd_rate": selected_node[2],
            "matched_by_description": bool(best_match),
            "available_subheadings": [
                {"hs_code": m[0], "description": m[1]} for m in leaf_nodes[:5]
            ],
            "transposed_from": transposed_from,
        }

    cursor.execute(
        "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code = ?",
        (clean_hs,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "status": "VALID_EXACT",
            "selected_hs_code": row[0],
            "description": row[1],
            "cd_rate": row[2],
            "transposed_from": transposed_from,
        }
    else:
        return {
            "status": "INVALID_CODE",
            "message": f"HS Code '{clean_hs}' does not exist in the Pakistan Customs Tariff database.",
        }


# ==========================================
# 2. LEGAL RAG RETRIEVAL
# ==========================================
def get_legal_context(query_text: str, n_results: int = 2) -> List[Dict[str, str]]:
    """Retrieves top N relevant legal chunks using the pure-Python SQLite fallback."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return []

    try:
        return query_legal_chunks(query_text, n_results=n_results)
    except Exception as e:
        print(f"⚠️ Legal retrieval warning: {e}")
        return []


# ==========================================
# 3. ORCHESTRATOR CORE PIPELINE
# ==========================================
def run_customs_orchestrator(
    user_query: str,
    hs_code: str = "",
    fob_usd: float = 0.0,
    item_description: str = "",
    ait_rate: float = 6.0,
    fed_rate: float = 0.0,
    freight_usd: float = 0.0,
    insurance_usd: float = 0.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Executes the 4-stage AI RAG pipeline with Exact, Semantic, and LLM Context Caching.
    """
    # Dynamic Environment Variable Fetching
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None

    # Params signature for math isolation
    param_signature = (
        hs_code.strip(),
        fob_usd,
        ait_rate,
        fed_rate,
        freight_usd,
        insurance_usd,
    )

    # --- LEVEL 1: EXACT HASH CACHE ---
    raw_key = f"{user_query.strip().lower()}|{hs_code.strip()}|{fob_usd}|{item_description.strip().lower()}|{ait_rate}|{fed_rate}|{freight_usd}|{insurance_usd}"
    exact_hash = hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    if use_cache and exact_hash in _EXACT_QUERY_CACHE:
        print("⚡ [EXACT CACHE HIT] Returning instant cached report.")
        cached_res = _EXACT_QUERY_CACHE[exact_hash].copy()
        cached_res["cache_type"] = "EXACT_HASH"
        return cached_res

    # --- LEVEL 2: SEMANTIC COSINE SIMILARITY CACHE ---
    if use_cache:
        tokens = _tokenize(f"{user_query} {item_description}")
        query_vec = _vectorize(tokens)

        for sem_entry in _SEMANTIC_CACHE:
            if sem_entry["param_signature"] == param_signature:
                sim = _cosine_similarity(query_vec, sem_entry["query_vec"])
                if sim >= 0.88:  # 88% similarity threshold
                    print(
                        f"🧠 [SEMANTIC CACHE HIT] Reusing previous report (Similarity: {sim:.2%})."
                    )
                    cached_res = sem_entry["result"].copy()
                    cached_res["cache_type"] = f"SEMANTIC ({sim:.1%})"
                    return cached_res

    # Stage 1: Validate & Resolve HS Code
    hs_resolution = resolve_hs_code(hs_code, item_description or user_query)

    if hs_resolution["status"] in ["NOT_FOUND", "INVALID_PREFIX", "INVALID_CODE"]:
        return {
            "error": True,
            "message": hs_resolution["message"],
            "resolution_data": hs_resolution,
        }

    target_hs = str(hs_resolution["selected_hs_code"])

    # Stage 2: Execute Layer 4 Duty Math
    duty_calculation = calculate_customs_duty(
        fob_usd=fob_usd,
        hs_code=target_hs,
        ait_rate=ait_rate,
        fed_rate=fed_rate,
        freight_usd=freight_usd,
        insurance_usd=insurance_usd,
    )

    # Stage 3: Retrieve Legal Context from ChromaDB
    rag_query = (
        f"{item_description} {user_query} import valuation restrictions exemption"
    )
    legal_snippets = get_legal_context(rag_query, n_results=2)

    # Stage 4: Synthesize Final Report with Gemini
    system_prompt = """
You are an expert Pakistan Customs & Trade Consultant AI. 
Analyze the provided duty calculation breakdown and legal context snippets to generate a concise, professional assessment report.

Formatting Guidelines:
- State the resolved HS code and official FBR tariff description.
- Present a clean, markdown-formatted duty breakdown table.
- Highlight applicable legal rules, exemptions, or valuation conditions retrieved from the legal context.
- Keep tone authoritative, clear, and scannable.
"""

    context_payload = {
        "hs_resolution": hs_resolution,
        "math_breakdown": duty_calculation,
        "legal_references": legal_snippets,
    }

    report_text = ""
    if client:
        try:
            user_prompt = f"User Query: {user_query}\n\nData Context:\n{json.dumps(context_payload, indent=2)}"
            model_name = "gemini-3.1-flash-lite"

            # Check for Explicit Gemini Context Cache
            cache_name = get_or_create_context_cache(client, model=model_name)

            if cache_name:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_name  # Reuses cached instructions
                    ),
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    ),
                )

            report_text = response.text or ""
        except Exception as e:
            report_text = f"⚠️ Gemini LLM Synthesis Error: {e}"
    else:
        report_text = f"⚠️ Gemini API Client not initialized. Checked GEMINI_API_KEY (Found: {bool(api_key)})."

    result_payload = {
        "error": False,
        "cache_type": "NONE",
        "resolved_hs": target_hs,
        "tariff_description": hs_resolution.get("description", ""),
        "transposed_from": hs_resolution.get("transposed_from"),
        "calculation": duty_calculation,
        "legal_snippets": legal_snippets,
        "ai_report": report_text,
    }

    # Store in Exact & Semantic Caches
    _EXACT_QUERY_CACHE[exact_hash] = result_payload
    _SEMANTIC_CACHE.append(
        {
            "query_vec": _vectorize(_tokenize(f"{user_query} {item_description}")),
            "param_signature": param_signature,
            "result": result_payload,
        }
    )

    return result_payload


# ==========================================
# TEST RUN
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("🚀 LAYER 5: MULTI-TIER CACHING DEMO")
    print("==================================================")

    # 1. Uncached Call
    start_1 = time.time()
    res1 = run_customs_orchestrator(
        user_query="I want to import a mini van",
        hs_code="8703",
        fob_usd=15000,
        item_description="Mini Van",
    )
    dur_1 = time.time() - start_1
    print(
        f"⏱️ Call 1 (Uncached): Executed in {dur_1:.2f}s | Cache: {res1.get('cache_type')}\n"
    )

    # 2. Exact Hash Cache Hit
    start_2 = time.time()
    res2 = run_customs_orchestrator(
        user_query="I want to import a mini van",
        hs_code="8703",
        fob_usd=15000,
        item_description="Mini Van",
    )
    dur_2 = time.time() - start_2
    print(
        f"⏱️ Call 2 (Exact Hit): Executed in {dur_2:.4f}s | Cache: {res2.get('cache_type')}\n"
    )

    # 3. Semantic Cache Hit (Differently phrased query, same parameters)
    start_3 = time.time()
    res3 = run_customs_orchestrator(
        user_query="Please calculate duties for bringing in a minivan vehicle",
        hs_code="8703",
        fob_usd=15000,
        item_description="Mini Van",
    )
    dur_3 = time.time() - start_3
    print(
        f"⏱️ Call 3 (Semantic Hit): Executed in {dur_3:.4f}s | Cache: {res3.get('cache_type')}\n"
    )