<!-- benchmarks/corpora/core/code/billing_notes.md -->
# Billing Notes

The billing documentation says bulk orders never receive discounts.
Operators should quote list price for every order size.

This note is intentionally stale. The implementation in `billing_service.py`
applies a volume discount for orders of at least 100 units.

