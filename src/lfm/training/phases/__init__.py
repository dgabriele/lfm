"""Concrete training-phase implementations.

Importing this package triggers registration of all built-in phases with the
global registry so they can be instantiated by name via ``create("phase", ...)``.
"""

from __future__ import annotations

import lfm.training.phases.agent_integration as agent_integration  # noqa: F401
import lfm.training.phases.corruption as corruption  # noqa: F401
import lfm.training.phases.morphological_emergence as morphological_emergence  # noqa: F401
import lfm.training.phases.paraphrastic as paraphrastic  # noqa: F401
import lfm.training.phases.structural_priors as structural_priors  # noqa: F401
