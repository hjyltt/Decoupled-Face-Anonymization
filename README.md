# Decoupled-Face-Anonymization
An experiment for Face Anonymization

## Project layout

```text
src/
  model/
  pipeline/
  util/
```

## Dependencies

Install the Python dependencies before running the code:

```bash
pip install -r requirements.txt
```

If you plan to use GPU inference, replace `onnxruntime` with the GPU build that
matches your local CUDA environment.

## InsightFace utility example

```python
from src.util import InsightFaceClient, InsightFaceConfig

config = InsightFaceConfig(
    model_name="buffalo_l",
    providers=["CPUExecutionProvider"],
    det_size=(640, 640),
)

client = InsightFaceClient(config=config)
faces = client.detect_faces("example.jpg")
largest_face = client.select_face(faces, strategy="largest")

if largest_face is not None:
    aligned_face = client.align_face("example.jpg", largest_face, image_size=112)
    embedding = largest_face.get_embedding(normalized=True)
```
