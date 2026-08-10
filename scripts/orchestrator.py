import sys
import os
import json
import sqlite3
import re
import hashlib
import time
import math
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple,cast

# Ensure local packages & paths are accessible
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".pythonlibs/lib/python3.11/site-packages"))

from google import genai
from google.genai import types
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Import Layer 4 Math Engine
from scripts.calculator import calculate_customs_duty, DB_PATH

VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "customs_legal_acts"
SYNONYMS_FILE_PATH = os.path.join("data", "consumer_synonyms.json")

# In-Memory Cache Containers for Orchestrator Pipeline
_EXACT_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_SEMANTIC_CACHE: List[Dict[str, Any]] = []
_SEMANTIC_CACHE_MAX_SIZE = 500  # FIX #3: bound cache growth (was unbounded -> memory leak)

# Global Gemini Context Cache handle
_GEMINI_CACHE_HANDLE: Optional[str] = None
_GEMINI_CACHE_EXPIRY: float = 0.0

# FIX #2: cache the parsed synonyms dict at module scope instead of re-reading
# the JSON file from disk on every single call to resolve_hs_code().
_SYNONYMS_CACHE: Optional[Tuple[Dict[str, List[str]], Dict[str, List[str]]]] = None


def load_consumer_synonyms() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Loads trade synonyms and direct PCT matches from JSON file.
    Supports both flat key-value dictionaries and hierarchical category dictionaries.
    Result is cached in-process after the first successful load (FIX #2).
    Returns:
      (synonyms_map, pct_matches_map)
    """
    global _SYNONYMS_CACHE
    if _SYNONYMS_CACHE is not None:
        return _SYNONYMS_CACHE

    synonyms_map: Dict[str, List[str]] = {}
    pct_matches_map: Dict[str, List[str]] = {}

    if os.path.exists(SYNONYMS_FILE_PATH):
        try:
            with open(SYNONYMS_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, val in data.items():
                if isinstance(val, dict):
                    # Check if flat term structure: "term": { "synonyms": [...], "pct_matches": [...] }
                    if "synonyms" in val:
                        term_lower = key.lower()
                        synonyms_map[term_lower] = [s.lower() for s in val.get("synonyms", [])]
                        pct_matches_map[term_lower] = [
                            m.get("hs_code") for m in val.get("pct_matches", []) if m.get("hs_code")
                        ]
                    else:
                        # Category-nested structure: "electronics": { "smartphone": { ... } }
                        for sub_term, term_data in val.items():
                            term_lower = sub_term.lower()
                            if isinstance(term_data, dict):
                                synonyms = [s.lower() for s in term_data.get("synonyms", [])]
                                synonyms_map[term_lower] = synonyms
                                hs_matches = [
                                    m.get("hs_code") for m in term_data.get("pct_matches", []) if m.get("hs_code")
                                ]
                                pct_matches_map[term_lower] = hs_matches

                                # Map individual synonyms back to term and HS matches
                                for syn in synonyms:
                                    if syn not in synonyms_map:
                                        synonyms_map[syn] = [term_lower]
                                    if syn not in pct_matches_map:
                                        pct_matches_map[syn] = hs_matches
                            elif isinstance(term_data, list):
                                synonyms_map[term_lower] = [s.lower() for s in term_data]

                elif isinstance(val, list):
                    synonyms_map[key.lower()] = [s.lower() for s in val]

            _SYNONYMS_CACHE = (synonyms_map, pct_matches_map)
            return _SYNONYMS_CACHE
        except Exception as e:
            print(f"⚠️ Notice: Could not load {SYNONYMS_FILE_PATH}: {e}")

    # Fallback default dictionary
    fallback_syns = {
        "smartphone": ["cellular", "mobile", "telephone"],
        "phone": ["cellular", "mobile", "telephone"],
        "laptop": ["automatic data processing", "portable", "computer"],
        "car": ["motor cars", "vehicles", "passenger"],
        "bull": ["bovine", "bovines", "animals"],
    }
    _SYNONYMS_CACHE = (fallback_syns, {})
    return _SYNONYMS_CACHE


class NativePythonEmbeddingFunction(EmbeddingFunction):
    """
    Pure-Python lightweight vector embedding function.
    Completely bypasses onnxruntime and C++ DLL dependencies on Windows/Mac/Linux.
    """
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 2]

    @staticmethod
    def _stable_hash(token: str) -> int:
        # FIX #1: Python's built-in hash() is randomized per-process for strings
        # (PYTHONHASHSEED) unless explicitly disabled. Because this embedding
        # function backs a *persistent* Chroma store, using hash() means every
        # process restart silently produces a different vector space than the
        # one documents were indexed with -- queries stop matching anything
        # meaningfully, with no error raised. hashlib.md5 is stable across runs.
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def _embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.vector_dim
        if not tokens:
            return vec
        for token in tokens:
            idx = self._stable_hash(token) % self.vector_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(doc) for doc in input]  # type: ignore[return-value]


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


def get_or_create_context_cache(
    client: genai.Client, model: str = "gemini-3.1-flash-lite"
) -> Optional[str]:
    """Creates or retrieves an explicit Gemini CachedContent resource."""
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
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                contents=base_instructions.strip(),
                ttl="3600s",
            ),
        )
        _GEMINI_CACHE_HANDLE = cache.name
        _GEMINI_CACHE_EXPIRY = current_time + 3500
        print(f"✅ Created Gemini Context Cache: {cache.name}")
        return _GEMINI_CACHE_HANDLE
    except Exception as e:
        print(f"ℹ️ Context Cache Notice: {e} (Falling back to standard prompt execution)")
        return None


def _best_scored_rows(
    cursor: sqlite3.Cursor, candidate_terms: List[str], keywords: List[str], limit: int = 5
) -> List[Tuple[str, str, float]]:
    """
    FIX #4: Search the DB for every candidate term (instead of stopping at the
    first term that returns *any* row), then rank all collected rows by how
    many of the original keywords actually appear in the description. This
    avoids returning an early, low-relevance match just because it happened
    to be searched first (e.g. "black leather jacket" no longer risks
    resolving on an unrelated heading that merely mentions "black").
    """
    seen: Dict[str, Tuple[str, str, float]] = {}
    for term in candidate_terms:
        query_term = f"%{term}%"
        cursor.execute(
            "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE description LIKE ? AND LENGTH(hs_code) = 8 LIMIT 20",
            (query_term,),
        )
        for hs, desc, cd in cursor.fetchall():
            if hs not in seen:
                seen[hs] = (hs, desc, cd)

    if not seen:
        return []

    scored = []
    for hs, desc, cd in seen.values():
        desc_lower = desc.lower()
        score = sum(1 for kw in keywords if kw in desc_lower)
        scored.append((score, hs, desc, cd))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(hs, desc, cd) for _, hs, desc, cd in scored[:limit]]


def resolve_hs_code(hs_input: str, item_description: str = "") -> Dict[str, Any]:
    """Validates, searches, and dynamically resolves HS codes from user input using multi-tier lookups."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    desc_clean = item_description.strip()
    clean_hs = re.sub(r"\D", "", hs_input)

    synonyms_dict, pct_matches_dict = load_consumer_synonyms()

    if not clean_hs:
        stopwords = {
            "i", "want", "to", "import", "an", "a", "the", "for", "my", "please",
            "calculate", "customs", "duty", "on", "bringing", "in", "from", "australian",
            "pakistan", "buying", "importing", "get", "rate"
        }
        all_words = [w.lower() for w in re.findall(r"\w+", desc_clean)]
        keywords = [w for w in all_words if w not in stopwords and len(w) > 2]

        matches = []

        # Tier 0: Direct Master PCT Matches from JSON dictionary
        matched_hs_codes = []
        for kw in keywords:
            if kw in pct_matches_dict and pct_matches_dict[kw]:
                matched_hs_codes.extend(pct_matches_dict[kw])

        if matched_hs_codes:
            for direct_hs in matched_hs_codes:
                clean_direct = direct_hs.replace(".", "").strip()
                if len(clean_direct) == 8:
                    cursor.execute(
                        "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code = ?",
                        (clean_direct,),
                    )
                else:
                    prefix = clean_direct[:4] + "%"
                    cursor.execute(
                        "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code LIKE ? AND LENGTH(hs_code) = 8 LIMIT 5",
                        (prefix,),
                    )
                res = cursor.fetchall()
                if res:
                    matches.extend(res)
                    break

        # Tier 1: SQL Keyword Search (FIX #4: score across all keywords, don't stop at first hit)
        if not matches and keywords:
            best_rows = _best_scored_rows(cursor, keywords, keywords)
            if best_rows:
                matches.extend(best_rows)

        # Tier 2: Synonym Expansion (FIX #4: same scoring approach)
        if not matches and keywords:
            expanded_terms = []
            for kw in keywords:
                if kw in synonyms_dict:
                    expanded_terms.extend(synonyms_dict[kw])

            if expanded_terms:
                best_rows = _best_scored_rows(cursor, expanded_terms, keywords)
                if best_rows:
                    matches.extend(best_rows)

        # Tier 3: Gemini AI Expansion Fallback
        if not matches and keywords:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    client = genai.Client(api_key=api_key)
                    ai_prompt = f"Given the user request '{item_description}', output ONLY the official 8-digit or 4-digit Harmonized System (HS) code prefix or technical tariff term (e.g., '8517' or 'cellular'). No markdown, no explanation."
                    response = client.models.generate_content(
                        model="gemini-3.1-flash-lite",
                        contents=ai_prompt
                    )
                    # Safely extract text from the AI response which may be None or under different attrs
                    raw_text = ""
                    if hasattr(response, "text") and response.text is not None:
                        raw_text = response.text
                    elif isinstance(response, dict):
                        # guard for dict-shaped responses
                        raw_text = response.get("text") or response.get("content") or ""

                    # ensure a str and normalize
                    if isinstance(raw_text, bytes):
                        try:
                            raw_text = raw_text.decode("utf-8", errors="ignore")
                        except Exception:
                            raw_text = ""

                    ai_suggestion = (raw_text or "").strip().replace(".", "")
                    ai_hs_digits = re.sub(r"\D", "", ai_suggestion)

                    if len(ai_hs_digits) >= 4:
                        prefix = ai_hs_digits[:4] + "%"
                        cursor.execute(
                            "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code LIKE ? AND LENGTH(hs_code) = 8 LIMIT 5",
                            (prefix,),
                        )
                        matches = cursor.fetchall()
                    elif ai_suggestion:
                        cursor.execute(
                            "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE description LIKE ? AND LENGTH(hs_code) = 8 LIMIT 5",
                            (f"%{ai_suggestion.lower()}%",),
                        )
                        matches = cursor.fetchall()
                except Exception as e:
                    print(f"⚠️ Gemini AI HS resolution fallback notice: {e}")

        conn.close()

        if matches:
            return {
                "status": "AUTO_RESOLVED",
                "selected_hs_code": matches[0][0],
                "description": matches[0][1],
                "cd_rate": matches[0][2],
                "alternatives": [
                    {"hs_code": m[0], "description": m[1]} for m in matches[1:]
                ],
            }
        else:
            return {
                "status": "NOT_FOUND",
                "message": f"Could not auto-detect HS code for '{item_description}'. Please specify key terms or an 8-digit HS Code.",
            }

    # Partial prefix expansion (< 8 digits)
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
            kw_list = [kw.lower() for kw in re.findall(r"\w+", desc_clean) if len(kw) > 2]
            highest_score = 0
            for node in leaf_nodes:
                node_hs, node_desc, node_cd = node
                node_desc_lower = node_desc.lower()
                score = sum(1 for kw in kw_list if kw in node_desc_lower)
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
        }

    # Exact 8-digit HS Code Lookup
    cursor.execute(
        "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE hs_code = ?",
        (clean_hs,),
    )
    row = cursor.fetchone()

    if row:
        conn.close()
        return {
            "status": "VALID_EXACT",
            "selected_hs_code": row[0],
            "description": row[1],
            "cd_rate": row[2],
        }
    else:
        # Fallback for invalid 8-digit code
        query_term = f"%{desc_clean}%"
        cursor.execute(
            "SELECT hs_code, description, cd_rate FROM pct_tariff WHERE description LIKE ? AND LENGTH(hs_code) = 8 LIMIT 1",
            (query_term,),
        )
        fallback_row = cursor.fetchone()
        conn.close()

        if fallback_row:
            return {
                "status": "INVALID_CODE_AUTO_CORRECTED",
                "message": f"Provided HS Code '{clean_hs}' was invalid. Auto-corrected to nearest tariff match '{fallback_row[0]}'.",
                "selected_hs_code": fallback_row[0],
                "description": fallback_row[1],
                "cd_rate": fallback_row[2],
            }
        else:
            return {
                "status": "INVALID_CODE",
                "message": f"HS Code '{clean_hs}' does not exist in the Pakistan Customs Tariff database.",
            }


def get_legal_context(query_text: str, n_results: int = 2) -> List[Dict[str, str]]:
    """Retrieves top N relevant legal chunks from ChromaDB vector store, with statutory legal fallbacks."""
    contexts: List[Dict[str, str]] = []

    if os.path.exists(VECTOR_STORE_PATH):
        try:
            embedding_fn = NativePythonEmbeddingFunction()
            chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
            collection = chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn
            )

            if collection.count() > 0:
                results = collection.query(query_texts=[query_text], n_results=n_results)
                docs = results.get("documents")
                metas = results.get("metadatas")

                if docs and metas and len(docs) > 0 and len(metas) > 0:
                    doc_list = docs[0]
                    meta_list = metas[0]
                    if doc_list and meta_list:
                        for doc, meta in zip(doc_list, meta_list):
                            source_name = (
                                meta.get("source", "Customs Legal Acts")
                                if isinstance(meta, dict)
                                else "Customs Legal Acts"
                            )
                            contexts.append({"source": str(source_name), "text": doc.strip()})
        except Exception as e:
            print(f"⚠️ ChromaDB Retrieval Warning: {e}")

    # Fallback statutory legal provisions
    if not contexts:
        contexts = [
            {
                "source": "Customs Act 1969 - Section 25 (Valuation of Imported Goods)",
                "text": "The value of imported goods shall be determined under Section 25 of the Customs Act, 1969 based on the transaction value, including freight, insurance, and landing charges (CIF standard valuation)."
            },
            {
                "source": "Customs Act 1969 - Section 80 (Checking of Goods Declaration)",
                "text": "Goods Declaration (GD) submitted through WeBOC/PSW shall be assessed for applicable customs duties, sales tax, federal excise duty, and income tax under the Pakistan Customs Tariff schedules."
            }
        ]

    return contexts


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
    """Executes the 4-stage AI RAG pipeline with Exact, Semantic, and Context Caching."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None

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
        cached_res["cache_hit"] = True
        return cached_res

    # --- LEVEL 2: SEMANTIC COSINE SIMILARITY CACHE ---
    if use_cache:
        tokens = _tokenize(f"{user_query} {item_description}")
        query_vec = _vectorize(tokens)

        for sem_entry in _SEMANTIC_CACHE:
            if sem_entry["param_signature"] == param_signature:
                sim = _cosine_similarity(query_vec, sem_entry["query_vec"])
                if sim >= 0.88:
                    print(f"🧠 [SEMANTIC CACHE HIT] Reusing previous report (Similarity: {sim:.2%}).")
                    cached_res = sem_entry["result"].copy()
                    cached_res["cache_type"] = f"SEMANTIC ({sim:.1%})"
                    cached_res["cache_hit"] = True
                    return cached_res

    hs_resolution = resolve_hs_code(hs_code, item_description or user_query)

    if hs_resolution["status"] in ["NOT_FOUND", "INVALID_PREFIX", "INVALID_CODE"]:
        return {
            "error": True,
            "message": hs_resolution["message"],
            "resolution_data": hs_resolution,
        }

    target_hs = str(hs_resolution["selected_hs_code"])

    duty_calculation = calculate_customs_duty(
        fob_usd=fob_usd,
        hs_code=target_hs,
        ait_rate=ait_rate,
        fed_rate=fed_rate,
        freight_usd=freight_usd,
        insurance_usd=insurance_usd,
    )

    rag_query = f"{item_description} {user_query} import valuation restrictions exemption"
    legal_snippets = get_legal_context(rag_query, n_results=2)

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

            cache_name = get_or_create_context_cache(client, model=model_name)

            if cache_name:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(cached_content=cache_name),
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                )

            report_text = response.text or ""
        except Exception as e:
            report_text = f"⚠️ Gemini LLM Synthesis Error: {e}"
    else:
        report_text = f"⚠️ Gemini API Client not initialized. Checked GEMINI_API_KEY (Found: {bool(api_key)})."

    result_payload = {
        "error": False,
        "cache_type": "NONE",
        "cache_hit": False,
        "hs_code": target_hs,
        "resolved_hs": target_hs,
        "tariff_description": hs_resolution.get("description", ""),
        "calculation": duty_calculation,
        "duty_calculation": duty_calculation,
        "legal_snippets": legal_snippets,
        "legal_context": [s["text"] for s in legal_snippets],
        "ai_report": report_text,
        "summary_report": report_text,
    }

    _EXACT_QUERY_CACHE[exact_hash] = result_payload

    # FIX #3: bound the semantic cache instead of growing it forever.
    if len(_SEMANTIC_CACHE) >= _SEMANTIC_CACHE_MAX_SIZE:
        _SEMANTIC_CACHE.pop(0)  # evict oldest entry (simple FIFO)

    _SEMANTIC_CACHE.append(
        {
            "query_vec": _vectorize(_tokenize(f"{user_query} {item_description}")),
            "param_signature": param_signature,
            "result": result_payload,
        }
    )

    return result_payload


if __name__ == "__main__":
    print("==================================================")
    print("🚀 LAYER 5: MULTI-TIER CACHING DEMO")
    print("==================================================")

    start_1 = time.time()
    res1 = run_customs_orchestrator(
        user_query="I want to import a smartphone",
        hs_code="",
        fob_usd=500,
        item_description="smartphone",
    )
    dur_1 = time.time() - start_1
    print(f"⏱️ Executed in {dur_1:.2f}s | Resolved HS: {res1.get('hs_code')}")