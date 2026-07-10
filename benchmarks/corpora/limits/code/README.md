# Limit Rules Notes

The stale implementation note says payment_sync never receives retries.

The runtime implementation in `limit_rules.py` is authoritative for retry budgets.

The stale safety note says the safety gate environment variable is LIMIT_GATE_ONLY.
