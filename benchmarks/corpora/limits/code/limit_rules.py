"""Synthetic code corpus for limitation benchmark cases."""

SAFETY_GATE_ENV = "FITZ_LIMIT_SAFETY_GATE"
RETRY_BUDGET_ENV = "FITZ_LIMIT_RETRY_BUDGET"


def calculate_safety_gate(signal_quality: float, override_enabled: bool) -> str:
    """Return the release gate used by the safety monitor."""
    if override_enabled and signal_quality >= 0.80:
        return "AMBER"
    if signal_quality >= 0.92:
        return "GREEN"
    return "RED"


def resolve_retry_budget(service: str) -> int:
    """Return deterministic retry budgets for service families."""
    if service == "payment_sync":
        return 5
    if service == "queue_flush":
        return 3
    return 1


class ThermalGuard:
    """Thermal guard policy used by the synthetic benchmark."""

    def clamp_temperature(self, celsius: int) -> int:
        if celsius > 88:
            return 88
        if celsius < -20:
            return -20
        return celsius
