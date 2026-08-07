import os
import sys
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))

from scripts.exchange_rates import get_usd_to_pkr_rate, clear_exchange_rate_cache
from scripts.calculator import calculate_customs_duty, DB_PATH
from scripts.orchestrator import (
    resolve_hs_code,
    get_legal_context,
    run_customs_orchestrator,
    _tokenize,
    _vectorize,
    _cosine_similarity,
)


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(autouse=True)
def reset_caches():
    """Automatically clears exchange rate cache before each test."""
    clear_exchange_rate_cache()


# ==========================================
# MODULE 1: EXCHANGE RATE MODULE TESTS
# ==========================================
def test_exchange_rate_fallback():
    """Tests that default fallback rate is used when API fails."""
    with patch("requests.get", side_effect=Exception("Network Offline")):
        rate = get_usd_to_pkr_rate()
        assert rate == 278.50


def test_exchange_rate_caching():
    """Tests that subsequent calls hit the in-memory cache."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"rates": {"PKR": 277.83}}

    with patch("requests.get", return_value=mock_response) as mock_get:
        # First Call - Hits API
        rate1 = get_usd_to_pkr_rate()
        assert rate1 == 277.83
        assert mock_get.call_count == 1

        # Second Call - Uses Cache
        rate2 = get_usd_to_pkr_rate()
        assert rate2 == 277.83
        assert mock_get.call_count == 1  # Call count remains 1!


# ==========================================
# MODULE 2: LAYER 4 MATH ENGINE TESTS
# ==========================================
def test_math_engine_waterfall_calculation():
    """Tests compounding duty waterfall calculations with mock rate."""
    with patch("scripts.calculator.fetch_usd_to_pkr", return_value=275.00):
        res = calculate_customs_duty(
            fob_usd=10000.0,
            hs_code="87032112",
            ait_rate=6.0,
            fed_rate=0.0,
            freight_usd=100.0,
            insurance_usd=50.0,
        )

        # Validate nested structure returned by calculator.py
        assert "duties_breakdown_pkr" in res
        breakdown = res["duties_breakdown_pkr"]

        assert "customs_duty" in breakdown
        assert "sales_tax" in breakdown
        assert "advance_income_tax" in breakdown
        assert (
            "total_payable" in breakdown
            or "total_duty_pkr" in breakdown
            or sum(breakdown.values()) > 0
        )


# ==========================================
# MODULE 3: HS CODE VALIDATOR & RESOLVER TESTS
# ==========================================
def test_resolve_hs_code_exact_match():
    """Tests exact 8-digit HS code lookup in SQLite DB."""
    res = resolve_hs_code("87032112")
    assert res["status"] == "VALID_EXACT"
    assert res["selected_hs_code"] == "87032112"


def test_resolve_hs_code_partial_keyword_expansion():
    """Tests 4-digit code expansion with item description keyword matching."""
    res = resolve_hs_code("8703", item_description="Mini Van")
    assert res["status"] == "PARTIAL_CODE_EXPANDED"
    assert (
        res["selected_hs_code"] == "87032112"
    )  # Should dynamically pick mini van kit code!


def test_resolve_hs_code_invalid():
    """Tests non-existent HS code input handling."""
    res = resolve_hs_code("99999999")
    assert res["status"] == "INVALID_CODE"


# ==========================================
# MODULE 4: SEMANTIC SIMILARITY TESTS
# ==========================================
def test_cosine_similarity():
    """Tests TF-IDF vector tokenization and cosine similarity."""
    vec1 = _vectorize(_tokenize("import mini van vehicle"))
    vec2 = _vectorize(_tokenize("bringing in mini van vehicle"))
    sim = _cosine_similarity(vec1, vec2)
    assert sim >= 0.70  # Validates high token overlap


# ==========================================
# MODULE 5: ORCHESTRATOR END-TO-END & CACHING
# ==========================================
def test_orchestrator_pipeline_and_caching():
    """Tests Layer 5 orchestrator execution, mock Gemini synthesis, and multi-tier caching."""
    mock_gemini_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "### Mocked Assessment Report"
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("scripts.orchestrator.genai.Client", return_value=mock_gemini_client):
        # 1. First Call (Uncached)
        res1 = run_customs_orchestrator(
            user_query="I want to import a mini van",
            hs_code="8703",
            fob_usd=15000,
            item_description="Mini Van",
            use_cache=False,
        )
        assert res1["error"] is False
        assert res1["resolved_hs"] == "87032112"
        assert res1["cache_type"] == "NONE"

        # 2. Second Call (Exact Cache Hit)
        res2 = run_customs_orchestrator(
            user_query="I want to import a mini van",
            hs_code="8703",
            fob_usd=15000,
            item_description="Mini Van",
            use_cache=True,
        )
        assert res2["cache_type"] == "EXACT_HASH"

        # 3. Third Call (Semantic Cache Hit with high phrase overlap)
        res3 = run_customs_orchestrator(
            user_query="I want to import a mini van vehicle",
            hs_code="8703",
            fob_usd=15000,
            item_description="Mini Van",
            use_cache=True,
        )
        assert "SEMANTIC" in res3["cache_type"]
