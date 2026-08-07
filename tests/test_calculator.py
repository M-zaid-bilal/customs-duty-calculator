import sys
import os
import unittest

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath("."))

from scripts.calculator import (
    calculate_customs_duty,
    get_customs_duty_rate,
    get_sindh_cess_rate,
)


class TestCustomsCalculator(unittest.TestCase):
    def test_live_animals_hs0101(self):
        """Test HS Code 0101 (Live Horses/Animals - 0% CD)."""
        res = calculate_customs_duty(
            fob_usd=5000,
            hs_code="0101",
            exchange_rate=280.0,  # Fixed rate for deterministic testing
        )
        # CIF = 5000 * 280 = 1,400,000
        # AV = 1,400,000 + 1% (14,000) = 1,414,000
        self.assertEqual(res["inputs"]["assessed_value_pkr"], 1414000.0)
        self.assertEqual(res["rates_applied"]["cd_rate"], 0.0)
        self.assertEqual(res["duties_breakdown_pkr"]["customs_duty"], 0.0)

    def test_freight_and_insurance_addition(self):
        """Test that Freight and Insurance are properly added to CIF prior to Landing Charges."""
        res = calculate_customs_duty(
            fob_usd=10000,
            freight_usd=500,
            insurance_usd=100,
            hs_code="0101",
            exchange_rate=280.0,
        )
        # CIF = (10000 + 500 + 100) * 280 = 2,968,000
        # AV = 2,968,000 * 1.01 = 2,997,680
        self.assertEqual(res["inputs"]["cif_pkr"], 2968000.0)
        self.assertEqual(res["inputs"]["assessed_value_pkr"], 2997680.0)

    def test_sindh_cess_slab_brackets(self):
        """Verify correct Sindh Cess percentage selection based on Assessed Value."""
        # Bracket 1: <= 1.25M PKR -> 1.80%
        self.assertEqual(get_sindh_cess_rate(1000000.0), 1.80)

        # Bracket 3: 2.03M to 4.06M PKR -> 1.82%
        self.assertEqual(get_sindh_cess_rate(3000000.0), 1.82)

        # Bracket 6: > 16M PKR -> 1.85%
        self.assertEqual(get_sindh_cess_rate(20000000.0), 1.85)

    def test_compounding_waterfall_math(self):
        """Verify the exact step-by-step compounding math for FED, Sales Tax, and AIT."""
        # Controlled Test Parameters
        av = 100000.0  # Assessed Value
        cd_rate = 10.0
        fed_rate = 5.0
        st_rate = 18.0
        ait_rate = 6.0

        # Calculations
        cd = av * 0.10  # 10,000
        fed = (av + cd) * 0.05  # (110,000) * 0.05 = 5,500
        st = (av + cd + fed) * 0.18  # (115,500) * 0.18 = 20,790
        ait = (av + cd + fed + st) * 0.06  # (136,290) * 0.06 = 8,177.40

        self.assertAlmostEqual(cd, 10000.0, places=2)
        self.assertAlmostEqual(fed, 5500.0, places=2)
        self.assertAlmostEqual(st, 20790.0, places=2)
        self.assertAlmostEqual(ait, 8177.40, places=2)

    def test_non_filer_ait_rate(self):
        """Test higher Advance Income Tax rate (12% for non-filers vs 6% standard)."""
        res_filer = calculate_customs_duty(
            fob_usd=1000, hs_code="0101", ait_rate=6.0, exchange_rate=280.0
        )
        res_non_filer = calculate_customs_duty(
            fob_usd=1000, hs_code="0101", ait_rate=12.0, exchange_rate=280.0
        )

        self.assertGreater(
            res_non_filer["duties_breakdown_pkr"]["advance_income_tax"],
            res_filer["duties_breakdown_pkr"]["advance_income_tax"],
        )


if __name__ == "__main__":
    unittest.main()
