from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
import os

import numpy as np

from .face_types import BoundingBox, FaceInfo

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import path only
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError as exc:  # pragma: no cover - import path only
    FaceAnalysis = None
    face_align = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None

ImageInput = str | Path | np.ndarray
FaceSortKey = Literal["score", "area", "left_to_right", "right_to_left"]
FaceSelectStrategy = Literal[
    "largest",
    "highest_score",
    "left_most",
    "right_most",
    "index",
]


@dataclass(slots=True)
class InsightFaceConfig:
    model_name: str = "buffalo_l"
    root: str | Path | None = None
    providers: list[str] | None = field(default_factory=lambda: ["CPUExecutionProvider"])
    allowed_modules: list[str] | None = None
    det_size: tuple[int, int] = (640, 640)
    det_thresh: float = 0.5
    ctx_id: int = 0


class InsightFaceClient:
    """Thin wrapper around insightface.app.FaceAnalysis for downstream reuse."""

    def __init__(self, config: InsightFaceConfig | None = None) -> None:
        self.config = config or InsightFaceConfig()
        self._require_runtime_dependencies()
        self.app = self._build_app()

    def detect_faces(
        self,
        image: ImageInput,
        max_num: int = 0,
        sort_by: FaceSortKey = "score",
    ) -> list[FaceInfo]:
        bgr_image = self.load_image(image)
        raw_faces = self._get_raw_faces(bgr_image=bgr_image, max_num=max_num)
        faces = [FaceInfo.from_insightface(face) for face in raw_faces]
        return self._sort_faces(faces, sort_by=sort_by)

    def count_faces(self, image: ImageInput, max_num: int = 0) -> int:
        return len(self.detect_faces(image=image, max_num=max_num))

    def get_face_boxes(
        self,
        image: ImageInput,
        max_num: int = 0,
        sort_by: FaceSortKey = "score",
    ) -> list[BoundingBox]:
        return [
            face.bbox
            for face in self.detect_faces(image=image, max_num=max_num, sort_by=sort_by)
        ]

    def select_face(
        self,
        faces: Sequence[FaceInfo],
        strategy: FaceSelectStrategy = "largest",
        index: int = 0,
    ) -> FaceInfo | None:
        if not faces:
            return None

        if strategy == "largest":
            return max(faces, key=lambda face: face.area)
        if strategy == "highest_score":
            return max(faces, key=lambda face: face.det_score)
        if strategy == "left_most":
            return min(faces, key=lambda face: face.bbox.x1)
        if strategy == "right_most":
            return max(faces, key=lambda face: face.bbox.x2)
        if strategy == "index":
            if index < 0 or index >= len(faces):
                return None
            return faces[index]
        raise ValueError(f"Unsupported face selection strategy: {strategy}")

    def detect_single_face(
        self,
        image: ImageInput,
        strategy: FaceSelectStrategy = "largest",
        index: int = 0,
    ) -> FaceInfo | None:
        faces = self.detect_faces(image=image)
        return self.select_face(faces=faces, strategy=strategy, index=index)

    def get_largest_face(self, image: ImageInput) -> FaceInfo | None:
        return self.detect_single_face(image=image, strategy="largest")

    def get_highest_score_face(self, image: ImageInput) -> FaceInfo | None:
        return self.detect_single_face(image=image, strategy="highest_score")

    def extract_embeddings(
        self,
        image: ImageInput,
        normalized: bool = True,
        max_num: int = 0,
        sort_by: FaceSortKey = "score",
    ) -> list[np.ndarray]:
        faces = self.detect_faces(image=image, max_num=max_num, sort_by=sort_by)
        embeddings: list[np.ndarray] = []
        for face in faces:
            embedding = face.get_embedding(normalized=normalized)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def crop_face(
        self,
        image: ImageInput,
        face: FaceInfo,
        margin: float = 0.0,
        square: bool = False,
    ) -> np.ndarray:
        if margin < 0:
            raise ValueError("margin must be non-negative.")

        bgr_image = self.load_image(image)
        bbox = face.bbox
        expanded_bbox = bbox.expanded(
            margin_x=bbox.width * margin,
            margin_y=bbox.height * margin,
        )
        if square:
            expanded_bbox = expanded_bbox.to_square()
        x1, y1, x2, y2 = expanded_bbox.clip(bgr_image.shape)
        return bgr_image[y1:y2, x1:x2].copy()

    def align_face(
        self,
        image: ImageInput,
        face: FaceInfo,
        image_size: int = 112,
    ) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("Face landmarks are required for alignment.")
        if face_align is None:
            raise ImportError("insightface.utils.face_align is unavailable.")

        bgr_image = self.load_image(image)
        return face_align.norm_crop(
            bgr_image,
            landmark=np.asarray(face.landmarks, dtype=np.float32),
            image_size=image_size,
        )

    def draw_detections(
        self,
        image: ImageInput,
        faces: Sequence[FaceInfo] | None = None,
        draw_landmarks: bool = True,
        draw_scores: bool = True,
        draw_demographics: bool = False,
    ) -> np.ndarray:
        if cv2 is None:
            raise ImportError("opencv-python is required to draw detections.")

        canvas = self.load_image(image).copy()
        faces = list(faces) if faces is not None else self.detect_faces(image)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.clip(canvas.shape)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            labels: list[str] = []
            if draw_scores:
                labels.append(f"score={face.det_score:.3f}")
            if draw_demographics:
                if face.gender is not None:
                    labels.append(f"gender={face.gender}")
                if face.age is not None:
                    labels.append(f"age={face.age}")
            if labels:
                cv2.putText(
                    canvas,
                    text=", ".join(labels),
                    org=(x1, max(0, y1 - 10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            if draw_landmarks and face.landmarks is not None:
                for x_coord, y_coord in np.asarray(face.landmarks, dtype=np.int32):
                    cv2.circle(canvas, (int(x_coord), int(y_coord)), 2, (0, 0, 255), -1)

        return canvas

    @staticmethod
    def load_image(image: ImageInput) -> np.ndarray:
        if cv2 is None:
            raise ImportError("opencv-python is required to load image inputs.")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image does not exist: {image_path}")
            file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise ValueError(f"Failed to decode image: {image_path}")
            return bgr_image

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a file path or a numpy.ndarray in BGR format.")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim != 3:
            raise ValueError("image must have 2 or 3 dimensions.")

        working_image = image
        if working_image.dtype != np.uint8:
            working_image = np.clip(working_image, 0, 255).astype(np.uint8)

        channels = working_image.shape[2]
        if channels == 3:
            return working_image.copy()
        if channels == 4:
            return cv2.cvtColor(working_image, cv2.COLOR_BGRA2BGR)

        raise ValueError("image must have 1, 3, or 4 channels.")

    def _build_app(self) -> FaceAnalysis:
        constructor_kwargs: dict[str, object] = {}
        if self.config.root is not None:
            constructor_kwargs["root"] = str(self.config.root)
        if self.config.providers is not None:
            constructor_kwargs["providers"] = list(self.config.providers)
        if self.config.allowed_modules is not None:
            constructor_kwargs["allowed_modules"] = list(self.config.allowed_modules)

        app = self._build_face_analysis_with_root_fallback(constructor_kwargs)

        try:
            app.prepare(
                ctx_id=self.config.ctx_id,
                det_thresh=self.config.det_thresh,
                det_size=self.config.det_size,
            )
        except TypeError:
            app.prepare(
                ctx_id=self.config.ctx_id,
                det_size=self.config.det_size,
            )
        return app

    def _build_face_analysis_with_root_fallback(
        self,
        constructor_kwargs: dict[str, object],
    ) -> FaceAnalysis:
        try:
            return self._build_face_analysis_with_fallbacks(constructor_kwargs)
        except PermissionError:
            if self.config.root is not None:
                raise

            fallback_root = self._default_local_model_root()
            fallback_root.mkdir(parents=True, exist_ok=True)
            fallback_kwargs = dict(constructor_kwargs)
            fallback_kwargs["root"] = str(fallback_root)
            return self._build_face_analysis_with_fallbacks(fallback_kwargs)

    def _build_face_analysis_with_fallbacks(
        self,
        constructor_kwargs: dict[str, object],
    ) -> FaceAnalysis:
        attempts = [
            constructor_kwargs,
            {
                key: value
                for key, value in constructor_kwargs.items()
                if key in {"root", "allowed_modules"}
            },
            {
                key: value
                for key, value in constructor_kwargs.items()
                if key == "root"
            },
            {},
        ]

        last_error: TypeError | None = None
        for attempt in attempts:
            try:
                return FaceAnalysis(name=self.config.model_name, **attempt)
            except TypeError as exc:
                last_error = exc

        raise RuntimeError("Failed to initialize FaceAnalysis.") from last_error

    def _get_raw_faces(self, bgr_image: np.ndarray, max_num: int) -> list[object]:
        try:
            return list(self.app.get(bgr_image, max_num=max_num))
        except TypeError:
            return list(self.app.get(bgr_image))

    @staticmethod
    def _sort_faces(faces: Sequence[FaceInfo], sort_by: FaceSortKey) -> list[FaceInfo]:
        if sort_by == "score":
            return sorted(faces, key=lambda face: face.det_score, reverse=True)
        if sort_by == "area":
            return sorted(faces, key=lambda face: face.area, reverse=True)
        if sort_by == "left_to_right":
            return sorted(faces, key=lambda face: face.bbox.x1)
        if sort_by == "right_to_left":
            return sorted(faces, key=lambda face: face.bbox.x1, reverse=True)
        raise ValueError(f"Unsupported face sorting rule: {sort_by}")

    @staticmethod
    def face_to_bbox(face: FaceInfo) -> BoundingBox:
        return face.bbox

    @staticmethod
    def _default_local_model_root() -> Path:
        return Path(os.getcwd()) / ".insightface"

    @staticmethod
    def _require_runtime_dependencies() -> None:
        missing: list[str] = []
        if _CV2_IMPORT_ERROR is not None:
            missing.append("opencv-python")
        if _INSIGHTFACE_IMPORT_ERROR is not None:
            missing.append("insightface")
        if missing:
            dependency_list = ", ".join(missing)
            raise ImportError(
                f"Missing runtime dependencies: {dependency_list}. "
                "Install packages from requirements.txt before using InsightFaceClient."
            )
