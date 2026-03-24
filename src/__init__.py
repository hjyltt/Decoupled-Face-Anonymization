"""Core package for face anonymization experiments."""

from .model import BaseAnonymizer
from .pipeline import BaseAnonymizationPipeline, PipelineOutput
from .util import BoundingBox, FaceInfo, InsightFaceClient, InsightFaceConfig

__all__ = [
    "BaseAnonymizationPipeline",
    "BaseAnonymizer",
    "BoundingBox",
    "FaceInfo",
    "InsightFaceClient",
    "InsightFaceConfig",
    "PipelineOutput",
]
