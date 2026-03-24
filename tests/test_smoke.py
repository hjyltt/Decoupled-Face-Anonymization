import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from src import (
    BaseAnonymizationPipeline,
    BaseAnonymizer,
    BoundingBox,
    FaceInfo,
    InsightFaceClient,
    InsightFaceConfig,
    PipelineOutput,
)


class DummyAnonymizer(BaseAnonymizer):
    def anonymize_face(self, image: np.ndarray, face: FaceInfo) -> np.ndarray:
        output = image.copy()
        x1, y1, x2, y2 = face.bbox.clip(output.shape)
        output[y1:y2, x1:x2] = 255
        return output


class DummyDetector:
    def __init__(self, faces: list[FaceInfo]) -> None:
        self._faces = faces

    def load_image(self, image: np.ndarray) -> np.ndarray:
        return image.copy()

    def detect_faces(self, image: np.ndarray) -> list[FaceInfo]:
        return list(self._faces)


class FakeCv2:
    COLOR_GRAY2BGR = 1
    COLOR_BGRA2BGR = 2

    @staticmethod
    def cvtColor(image: np.ndarray, code: int) -> np.ndarray:
        if code == FakeCv2.COLOR_GRAY2BGR:
            return np.repeat(image[:, :, None], 3, axis=2)
        if code == FakeCv2.COLOR_BGRA2BGR:
            return image[:, :, :3].copy()
        raise ValueError(f"Unsupported conversion code: {code}")


class SmokeTests(unittest.TestCase):
    def test_root_exports_are_available(self) -> None:
        self.assertIsNotNone(BaseAnonymizationPipeline)
        self.assertIsNotNone(BaseAnonymizer)
        self.assertIsNotNone(BoundingBox)
        self.assertIsNotNone(FaceInfo)
        self.assertIsNotNone(InsightFaceClient)
        self.assertIsNotNone(InsightFaceConfig)
        self.assertIsNotNone(PipelineOutput)

    def test_face_info_from_mapping_copies_embedding_data(self) -> None:
        source = {
            "bbox": [1, 2, 11, 14],
            "det_score": 0.95,
            "kps": [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
            "embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "normed_embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "gender": 1,
            "age": 28,
        }

        face = FaceInfo.from_insightface(source)
        source["embedding"][0] = 99.0

        self.assertEqual(face.bbox.as_int_tuple(), (1, 2, 11, 14))
        self.assertAlmostEqual(face.det_score, 0.95)
        self.assertEqual(face.gender, 1)
        self.assertEqual(face.age, 28)
        np.testing.assert_allclose(
            face.get_embedding(normalized=False),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_pipeline_run_returns_faces_and_processed_image(self) -> None:
        face = FaceInfo(bbox=BoundingBox(1, 1, 3, 3), det_score=0.9)
        detector = DummyDetector([face])
        pipeline = BaseAnonymizationPipeline(detector=detector, anonymizer=DummyAnonymizer())
        image = np.zeros((5, 5, 3), dtype=np.uint8)

        result = pipeline.run(image)

        self.assertIsInstance(result, PipelineOutput)
        self.assertEqual(len(result.faces), 1)
        np.testing.assert_array_equal(result.image[1:3, 1:3], np.full((2, 2, 3), 255))

    def test_select_face_supports_common_strategies(self) -> None:
        client = object.__new__(InsightFaceClient)
        faces = [
            FaceInfo(bbox=BoundingBox(0, 0, 4, 4), det_score=0.8),
            FaceInfo(bbox=BoundingBox(10, 0, 12, 2), det_score=0.95),
        ]

        self.assertEqual(client.select_face(faces, strategy="largest"), faces[0])
        self.assertEqual(client.select_face(faces, strategy="highest_score"), faces[1])
        self.assertEqual(client.select_face(faces, strategy="left_most"), faces[0])
        self.assertEqual(client.select_face(faces, strategy="right_most"), faces[1])
        self.assertEqual(client.select_face(faces, strategy="index", index=1), faces[1])
        self.assertIsNone(client.select_face(faces, strategy="index", index=5))

    def test_load_image_handles_array_inputs_with_opencv_available(self) -> None:
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 0] = 10
        rgba[:, :, 1] = 20
        rgba[:, :, 2] = 30
        rgba[:, :, 3] = 255

        with patch("src.util.insightface_client.cv2", new=FakeCv2()):
            gray_bgr = InsightFaceClient.load_image(gray)
            rgba_bgr = InsightFaceClient.load_image(rgba)

        self.assertEqual(gray_bgr.shape, (4, 4, 3))
        self.assertEqual(rgba_bgr.shape, (4, 4, 3))
        np.testing.assert_array_equal(rgba_bgr[0, 0], np.array([10, 20, 30], dtype=np.uint8))

    def test_build_app_falls_back_to_local_root_after_permission_error(self) -> None:
        client = object.__new__(InsightFaceClient)
        client.config = InsightFaceConfig(providers=["CPUExecutionProvider"])

        captured_kwargs: list[dict[str, object]] = []

        def fake_builder(kwargs: dict[str, object]) -> str:
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                raise PermissionError("default home directory is not writable")
            return "fake-app"

        with patch.object(client, "_build_face_analysis_with_fallbacks", side_effect=fake_builder):
            with patch.object(client, "_default_local_model_root", return_value=Path("fallback-root")):
                app = client._build_face_analysis_with_root_fallback({"providers": ["CPUExecutionProvider"]})

        self.assertEqual(app, "fake-app")
        self.assertEqual(captured_kwargs[0], {"providers": ["CPUExecutionProvider"]})
        self.assertEqual(
            captured_kwargs[1],
            {
                "providers": ["CPUExecutionProvider"],
                "root": "fallback-root",
            },
        )


if __name__ == "__main__":
    unittest.main()
