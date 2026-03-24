from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from src.util.face_types import FaceInfo


class BaseAnonymizer(ABC):
    """Abstract interface for all face anonymization models."""

    @abstractmethod
    def anonymize_face(self, image: np.ndarray, face: FaceInfo) -> np.ndarray:
        """Anonymize a single detected face region and return the updated image."""

    def anonymize_faces(self, image: np.ndarray, faces: Sequence[FaceInfo]) -> np.ndarray:
        output = image.copy()
        for face in faces:
            output = self.anonymize_face(output, face)
        return output
