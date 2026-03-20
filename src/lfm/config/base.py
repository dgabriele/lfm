"""Base configuration model for the LFM framework.

All LFM configuration classes should inherit from ``LFMBaseConfig`` to get
consistent validation behaviour: frozen (immutable) instances, forbidden extra
fields, and automatic enum-value coercion.
"""

from __future__ import annotations

from pydantic import BaseModel


class LFMBaseConfig(BaseModel):
    """Immutable, strictly-validated base for all LFM configs.

    Model configuration:
        - ``frozen = True`` — instances are immutable after creation.
        - ``extra = "forbid"`` — unknown fields raise a validation error.
        - ``use_enum_values = True`` — enum fields store their value, not the
          enum member, enabling cleaner serialization.
    """

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "use_enum_values": True,
    }
