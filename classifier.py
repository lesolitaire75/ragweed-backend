import io
import numpy as np
from PIL import Image
import onnxruntime as ort

CLASS_NAMES = ["ambrosia", "non_ambrosia"]
AMBROSIA_IDX = 0

print("[Classifier] Chargement du modèle ONNX...")
_session = ort.InferenceSession(
    "weights/best.onnx",
    providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name
print(f"[Classifier] Modèle prêt. Classes: {CLASS_NAMES}")


def predict(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = _session.run(None, {_input_name: arr})
    logits = outputs[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": round(top_conf, 4),
        "is_ragweed": top_idx == AMBROSIA_IDX,
        "all_scores": {
            "ambrosia": round(float(probs[AMBROSIA_IDX]), 4),
            "non_ambrosia": round(float(probs[1]), 4),
        }
    }
