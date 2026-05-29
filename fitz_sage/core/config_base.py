# fitz_sage/core/config_base.py
"""Shared config validation mixin.

``ValidationMixin`` runs ``validate_config()`` automatically after model
initialization; subclasses override ``validate_config()`` to add invariants.
"""

from __future__ import annotations

from pydantic import BaseModel


class ValidationMixin(BaseModel):
    """
    Base class for configs that need validation.

    Provides consistent validation pattern.
    """

    def validate_config(self) -> None:
        """
        Validate configuration consistency.

        Override in subclasses to add specific validation logic.
        Raises ValueError if config is invalid.
        """
        pass

    def model_post_init(self, __context) -> None:
        """Automatically run validation after initialization."""
        self.validate_config()
