import json
import urllib.request


def fetch_usd_to_pkr() -> float:
    """
    Fetches the live USD to PKR exchange rate using multiple free APIs with fallback logic.
    """
    # List of reliable free currency APIs
    apis = [
        # Provider 1: Open Exchange Rates (Free endpoint)
        {"url": "https://open.er-api.com/v6/latest/USD", "key_path": ["rates", "PKR"]},
        # Provider 2: ExchangeRate-API
        {
            "url": "https://api.exchangerate-api.com/v4/latest/USD",
            "key_path": ["rates", "PKR"],
        },
        # Provider 3: FrankFurter API
        {
            "url": "https://api.frankfurter.app/latest?from=USD&to=PKR",
            "key_path": ["rates", "PKR"],
        },
    ]

    for api in apis:
        try:
            req = urllib.request.Request(
                api["url"],  # ty:ignore[invalid-argument-type]
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode())

                # Navigate nested key paths dynamically
                rate = data
                for key in api["key_path"]:
                    rate = rate[key]

                rate = float(rate)
                print(
                    f"✅ Successfully fetched exchange rate from {api['url']}: 1 USD = {rate:.2f} PKR"
                )
                return rate

        except Exception as e:
            print(f"⚠️ Failed to fetch from {api['url']} ({e}). Trying next provider...")
            continue

    # Final hardcoded safety fallback if internet/APIs are completely down
    fallback_rate = 278.50
    print(
        f"❌ All exchange rate APIs failed. Utilizing static fallback rate: {fallback_rate} PKR"
    )
    return fallback_rate


if __name__ == "__main__":
    rate = fetch_usd_to_pkr()
    print(f"\nFinal Usable Exchange Rate: {rate} PKR")
