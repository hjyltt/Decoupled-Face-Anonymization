from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.base_anonymizer import BaseAnonymizer
from src.util.face_types import FaceInfo
from src.util.insightface_client import ImageInput, InsightFaceClient


@dataclass(slots=True)
class PipelineOutput:
    image: np.ndarray
    faces: list[FaceInfo]


class BaseAnonymizationPipeline:
    """Reference pipeline that wires face detection to an anonymization model."""

    def __init__(self, detector: InsightFaceClient, anonymizer: BaseAnonymizer) -> None:
        self.detector = detector
        self.anonymizer = anonymizer

    def run(self, image: ImageInput) -> PipelineOutput:
        bgr_image = self.detector.load_image(image)
        faces = self.detector.detect_faces(bgr_image)
        anonymized = self.anonymizer.anonymize_faces(image=bgr_image, faces=faces)
        return PipelineOutput(image=anonymized, faces=faces)
