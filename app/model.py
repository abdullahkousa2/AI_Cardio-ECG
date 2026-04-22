import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import cv2
import base64
import io
from PIL import Image

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_efficientnetv2s.pth"

CLASS_NAMES = [
    "Myocardial Infarction",
    "History of MI",
    "Abnormal Heartbeat",
    "Normal",
]

IMG_SIZE = (300, 300)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = np.array([0.5, 0.5, 0.5], dtype="float32")
_STD  = np.array([0.5, 0.5, 0.5], dtype="float32")


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        return x * self.fc(s).view(b, c, 1, 1)


class ECGModel(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.backbone.num_features  # 1280
        self.se  = SEBlock(feat_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.se(self.backbone(x))
        return self.classifier(self.gap(x).flatten(1) + self.gmp(x).flatten(1))


_model = None


def get_model():
    global _model
    if _model is None:
        m = ECGModel()
        state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.to(DEVICE).eval()
        _model = m
    return _model


def preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = (np.array(img, dtype="float32") / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _encode_img(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_gradcam(model: ECGModel, img_tensor: torch.Tensor, pred_idx: int) -> np.ndarray:
    _acts, _grads = [None], [None]

    def save_act(m, inp, out):
        _acts[0] = out

    def save_grad(m, grad_in, grad_out):
        _grads[0] = grad_out[0]

    fh = model.se.register_forward_hook(save_act)
    bh = model.se.register_full_backward_hook(save_grad)

    model.eval()
    img_tensor = img_tensor.detach().requires_grad_(True)
    logits = model(img_tensor)
    model.zero_grad()
    logits[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    acts  = _acts[0].detach()
    grads = _grads[0].detach() if _grads[0] is not None else torch.ones_like(acts)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1)).squeeze(0).cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()

    h, w = IMG_SIZE
    cam_up  = cv2.resize(cam, (w, h))
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    orig = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    orig = np.clip(orig * _STD + _MEAN, 0.0, 1.0)
    orig = np.uint8(orig * 255)

    return np.uint8(0.5 * orig + 0.5 * heatmap)


def predict(image_bytes: bytes) -> dict:
    model = get_model()

    with torch.no_grad():
        probs = torch.softmax(model(preprocess(image_bytes)), dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    gradcam_arr = compute_gradcam(model, preprocess(image_bytes), pred_idx)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "original_image": base64.b64encode(image_bytes).decode(),
        "gradcam_image": _encode_img(gradcam_arr),
    }
