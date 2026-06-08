# benchmarks/corpora/holdout/code/feature_flag_service.py
"""Feature flag service used by the holdout benchmark."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

KILL_SWITCH_ENV = "FITZ_HOLDOUT_KILL_SWITCH"


@dataclass(frozen=True)
class FeatureFlag:
    """Feature flag definition."""

    key: str
    region: str
    rollout_percent: int


class FlagEvaluator:
    """Evaluates feature flag eligibility."""

    def is_eligible(self, user: dict[str, str], flag: FeatureFlag) -> bool:
        """Return whether a user should see a feature flag."""
        if user.get("archived") == "true":
            return False
        if os.environ.get(KILL_SWITCH_ENV):
            return False
        if user.get("region") != flag.region:
            return False
        return self.rollout_bucket(user["id"], flag.key) < flag.rollout_percent

    def rollout_bucket(self, user_id: str, flag_key: str) -> int:
        """Return a stable 0-99 rollout bucket."""
        digest = hashlib.sha256(f"{user_id}:{flag_key}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100
