# benchmarks/corpora/core/code/billing_service.py
"""Billing service used by benchmark cases."""

from decimal import Decimal

TAX_RATE = Decimal("0.19")


class InvoiceCalculator:
    """Calculate invoice totals for benchmark orders."""

    def apply_volume_discount(self, subtotal: Decimal, quantity: int) -> Decimal:
        """Apply a ten percent discount for orders of at least 100 units."""
        if quantity >= 100:
            return subtotal * Decimal("0.90")
        return subtotal

    def calculate_total(self, subtotal: Decimal, quantity: int) -> Decimal:
        """Calculate the final invoice total after discount and tax."""
        discounted = self.apply_volume_discount(subtotal, quantity)
        return discounted * (Decimal("1.0") + TAX_RATE)
