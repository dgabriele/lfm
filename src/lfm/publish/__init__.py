"""Publishing infrastructure for HuggingFace Hub.

Reusable OOP framework for uploading models and datasets to HuggingFace,
with release manifest generation.

Public API::

    from lfm.publish import (
        HFPublisher,
        ModelRelease,
        DatasetRelease,
        ReleaseManifest,
    )
"""

from lfm.publish.base import HFPublisher, ReleaseManifest
from lfm.publish.dataset import DatasetRelease
from lfm.publish.model import ModelRelease

__all__ = [
    "HFPublisher",
    "ReleaseManifest",
    "ModelRelease",
    "DatasetRelease",
]
