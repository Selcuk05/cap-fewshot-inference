"""
Siamese / Prototypical encoder used by Trainer, plus loaders for .pt / ONNX / TorchScript.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def build_trunk(backbone, pretrained=False):
    weights = "DEFAULT" if pretrained else None
    if backbone.startswith("resnet"):
        model = getattr(models, backbone)(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    if backbone == "mobilenetV2":
        model = models.mobilenet_v2(weights=weights)
        feat_dim = model.last_channel
        model.classifier = nn.Identity()
        return model, feat_dim
    raise ValueError(f"Unsupported few-shot backbone: {backbone}")


class SiameseEncoder(nn.Module):
    def __init__(self, backbone="resnet18", embedding_dim=128, pretrained=False):
        super().__init__()
        self.backbone, feat_dim = build_trunk(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.head(features)


def query_scores(query, references, metric):
    if metric == "cosine":
        query = F.normalize(query, dim=1)
        references = F.normalize(references, dim=1)
        return query @ references.T
    query_sq = (query ** 2).sum(1, keepdim=True)
    ref_sq = (references ** 2).sum(1).unsqueeze(0)
    return -(query_sq - 2 * query @ references.T + ref_sq)


def scores_to_confidence(scores, metric):
    if metric == "cosine":
        return scores.clamp(0, 1)
    distance = (-scores).clamp(min=0).sqrt()
    return 1.0 / (1.0 + distance)


class TorchEncoder:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def embed(self, batch):
        return self.model(batch.to(self.device)).float()


class OnnxEncoder:
    def __init__(self, path, device):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = device

    def embed(self, batch):
        outputs = self.session.run(
            None, {self.input_name: batch.detach().cpu().numpy()}
        )
        return torch.from_numpy(outputs[0]).float()


def _is_encoder_state(state):
    return any(
        key.startswith("backbone.") or key.startswith("head.") for key in state
    )


def _encoder_from_checkpoint(payload, backbone, embedding_dim, device):
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not _is_encoder_state(state):
        raise ValueError("Checkpoint does not contain a few-shot encoder state_dict.")

    meta_backbone = payload.get("backbone") or backbone or "resnet18"
    meta_dim = int(payload.get("embedding_dim") or embedding_dim or 128)
    model = SiameseEncoder(meta_backbone, meta_dim, pretrained=False)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    meta = {
        "backbone": meta_backbone,
        "embedding_dim": meta_dim,
        "distance": payload.get("distance"),
        "architecture": payload.get("architecture"),
        "classes": payload.get("classes"),
    }
    return TorchEncoder(model, device), meta


def load_encoder(path, backbone, embedding_dim, device):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return OnnxEncoder(path, device), {}
    if suffix == ".torchscript":
        model = torch.jit.load(str(path), map_location=str(device))
        model.eval()
        return TorchEncoder(model, device), {}

    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception:
        payload = None

    if isinstance(payload, dict) and (
        "state_dict" in payload or _is_encoder_state(payload)
    ):
        return _encoder_from_checkpoint(payload, backbone, embedding_dim, device)
    if isinstance(payload, nn.Module):
        payload.to(device)
        payload.eval()
        return TorchEncoder(payload, device), {}

    model = torch.jit.load(str(path), map_location=str(device))
    model.eval()
    return TorchEncoder(model, device), {}
