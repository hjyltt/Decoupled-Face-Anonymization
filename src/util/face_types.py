from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _get_face_field(face: Any, name: str) -> Any:
    if hasattr(face, name):
        return getattr(face, name)
    if isinstance(face, dict):
        return face.get(name)
    try:
        return face[name]
    except (KeyError, IndexError, TypeError):
        return None


def _to_numpy_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value).copy()


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_array(cls, bbox: Any) -> "BoundingBox":
        bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_array.size < 4:
            raise ValueError("Bounding box must contain at least four values.")
        return cls(
            x1=float(bbox_array[0]),
            y1=float(bbox_array[1]),
            x2=float(bbox_array[2]),
            y2=float(bbox_array[3]),
        )

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def expanded(self, margin_x: float, margin_y: float) -> "BoundingBox":
        return BoundingBox(
            x1=self.x1 - margin_x,
            y1=self.y1 - margin_y,
            x2=self.x2 + margin_x,
            y2=self.y2 + margin_y,
        )

    def to_square(self) -> "BoundingBox":
        center_x, center_y = self.center
        half_side = max(self.width, self.height) / 2.0
        return BoundingBox(
            x1=center_x - half_side,
            y1=center_y - half_side,
            x2=center_x + half_side,
            y2=center_y + half_side,
        )

    def clip(self, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Image shape must describe a non-empty image.")

        x1 = max(0, min(width - 1, int(np.floor(self.x1))))
        y1 = max(0, min(height - 1, int(np.floor(self.y1))))
        x2 = max(x1 + 1, min(width, int(np.ceil(self.x2))))
        y2 = max(y1 + 1, min(height, int(np.ceil(self.y2))))
        return x1, y1, x2, y2

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


@dataclass(frozen=True)
class FaceInfo:
    bbox: BoundingBox
    det_score: float
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None
    normed_embedding: np.ndarray | None = None
    gender: int | None = None
    age: int | None = None

    @classmethod
    def from_insightface(cls, face: Any) -> "FaceInfo":
        raw_bbox = _get_face_field(face, "bbox")
        if raw_bbox is None:
            raise ValueError("InsightFace face object does not contain 'bbox'.")

        det_score = _get_face_field(face, "det_score")
        if det_score is None:
            det_score = _get_face_field(face, "score")

        return cls(
            bbox=BoundingBox.from_array(raw_bbox),
            det_score=float(det_score or 0.0),
            landmarks=_to_numpy_array(_get_face_field(face, "kps")),
            embedding=_to_numpy_array(_get_face_field(face, "embedding")),
            normed_embedding=_to_numpy_array(_get_face_field(face, "normed_embedding")),
            gender=_coerce_optional_int(_get_face_field(face, "gender")),
            age=_coerce_optional_int(_get_face_field(face, "age")),
        )

    @property
    def area(self) -> float:
        return self.bbox.area

    def get_embedding(self, normalized: bool = True) -> np.ndarray | None:
        if normalized and self.normed_embedding is not None:
            return self.normed_embedding.copy()
        if self.embedding is not None:
            return self.embedding.copy()
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.as_int_tuple(),
            "det_score": self.det_score,
            "landmarks": None if self.landmarks is None else self.landmarks.copy(),
            "embedding": None if self.embedding is None else self.embedding.copy(),
            "normed_embedding": None
            if self.normed_embedding is None
            else self.normed_embedding.copy(),
            "gender": self.gender,
            "age": self.age,
        }


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
