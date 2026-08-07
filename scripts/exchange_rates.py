import os
import time

# Safely handle optional requests module without triggering static type checker warnings
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    HAS_REQUESTS = False

# Default USD/PKR fallback rate if offline or API fails
DEFAULT_FALLBACK_RATE = 278.50

# Global In-Memory Cache
_RATE_CACHE = {"rate": None, "timestamp": 0.0}

# 12 Hours TTL (12 hours * 3600 seconds)
CACHE_TTL_SECONDS = 12 * 3600


def get_usd_to_pkr_rate(cache_ttl: int = CACHE_TTL_SECONDS) -> float:
    """
    Fetches the current USD to PKR exchange rate using Open Exchange Rates API.
    Caches the result in memory for `cache_ttl` seconds (default 12 hours).
    Falls back to environment variable or DEFAULT_FALLBACK_RATE if offline.
    """
    current_time = time.time()

    # 1. Check In-Memory Cache
    cached_rate = _RATE_CACHE["rate"]
    cached_time = _RATE_CACHE["timestamp"]

    if isinstance(cached_rate, float) and isinstance(cached_time, float):
        if (current_time - cached_time) < cache_ttl:
            return cached_rate

    # 2. Check Environment Variable Override (e.g. USD_PKR_RATE=280.0)
    env_rate = os.getenv("USD_PKR_RATE")
    if env_rate:
        try:
            rate = float(env_rate)
            _RATE_CACHE["rate"] = rate
            _RATE_CACHE["timestamp"] = current_time
            print(f"✅ Using custom environment exchange rate: 1 USD = {rate} PKR")
            return rate
        except ValueError:
            pass

    # 3. Fetch Live Exchange Rate via API
    if HAS_REQUESTS and requests is not None:
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                rate = float(data["rates"]["PKR"])

                # Save to Cache
                _RATE_CACHE["rate"] = rate
                _RATE_CACHE["timestamp"] = current_time
                print(
                    f"✅ Successfully fetched & cached exchange rate: 1 USD = {rate} PKR"
                )
                return rate
        except Exception as e:
            print(f"⚠️ Exchange Rate API failed ({e}). Falling back to default rate.")

    # 4. Fallback if offline or API fails
    _RATE_CACHE["rate"] = DEFAULT_FALLBACK_RATE
    _RATE_CACHE["timestamp"] = current_time
    print(f"ℹ️ Using fallback exchange rate: 1 USD = {DEFAULT_FALLBACK_RATE} PKR")
    return DEFAULT_FALLBACK_RATE


def clear_exchange_rate_cache() -> None:
    """Utility function to manually clear or invalidate the rate cache."""
    global _RATE_CACHE
    _RATE_CACHE = {"rate": None, "timestamp": 0.0}


if __name__ == "__main__":
    print("==================================================")
    print("🧪 TESTING EXCHANGE RATE MODULE (WITH CACHING)")
    print("==================================================")

    # First Call - Uncached (Triggers API call)
    start_time = time.time()
    rate_1 = get_usd_to_pkr_rate()
    time_1 = time.time() - start_time
    print(f"Result 1: 1 USD = {rate_1} PKR (took {time_1:.4f}s)")

    # Second Call - Instant Cache Hit
    start_time = time.time()
    rate_2 = get_usd_to_pkr_rate()
    time_2 = time.time() - start_time
    print(f"Result 2: 1 USD = {rate_2} PKR (took {time_2:.6f}s) [CACHE HIT]")
