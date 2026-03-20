"""Faculty subsystem for the LFM framework.

Provides the ``LanguageFaculty`` compositor and its ``FacultyConfig``.
The ``LanguageFaculty`` wires all LFM pipeline stages into a single
coherent forward pass.
"""

from __future__ import annotations

from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty

__all__ = ["FacultyConfig", "LanguageFaculty"]
