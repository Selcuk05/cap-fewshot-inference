from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from capsules.Fewshot.src.classes.encoder import (
    query_scores,
    scores_to_confidence,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _list_class_dirs(root):
    root = Path(root)
    return sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def _unwrap_root(extract_dir):
    extract_dir = Path(extract_dir)
    children = [
        path for path in extract_dir.iterdir() if path.name not in {".ready", "__MACOSX"}
    ]
    dirs = [path for path in children if path.is_dir()]
    files = [path for path in children if path.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extract_dir


def collect_gallery(extract_dir):
    root = _unwrap_root(extract_dir)
    train_dir = root / "train"
    if train_dir.is_dir() and _list_class_dirs(train_dir):
        class_dirs = _list_class_dirs(train_dir)
        samples, classes = _collect_images(class_dirs)
        val_dir = root / "val"
        if val_dir.is_dir():
            val_map = {path.name: path for path in _list_class_dirs(val_dir)}
            for name in classes:
                class_dir = val_map.get(name)
                if not class_dir:
                    continue
                label = classes.index(name)
                for path in sorted(class_dir.rglob("*")):
                    if _is_image(path):
                        samples.append((path, label))
        return samples, classes

    class_dirs = _list_class_dirs(root)
    if not class_dirs:
        raise FileNotFoundError(
            "Gallery zip must contain train/<identity>/ folders or class folders at the root."
        )
    return _collect_images(class_dirs)


def _collect_images(class_dirs):
    samples = []
    classes = []
    for index, class_dir in enumerate(class_dirs):
        classes.append(class_dir.name)
        for path in sorted(class_dir.rglob("*")):
            if _is_image(path):
                samples.append((path, index))
    return samples, classes


def image_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _to_uint8_rgb(image):
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Empty image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_bgr(image, image_size):
    rgb = _to_uint8_rgb(image)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _load_rgb_tensor(path, image_size):
    return image_transform(image_size)(Image.open(path).convert("RGB"))


@torch.no_grad()
def embed_paths(encoder, samples, image_size, batch_size=16):
    tensors = [_load_rgb_tensor(path, image_size) for path, _ in samples]
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    embeddings = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size])
        embeddings.append(encoder.embed(batch).cpu())
    return torch.cat(embeddings, dim=0), labels


def class_prototypes(embeddings, labels, class_count):
    prototypes = []
    for index in range(class_count):
        members = embeddings[labels == index]
        if members.numel() == 0:
            raise ValueError(f"Gallery class {index} has no embeddings.")
        prototypes.append(members.mean(0))
    return torch.stack(prototypes, dim=0)


def match_query(query, references, labels, class_count, metric):
    scores = query_scores(query, references.to(query.device), metric)[0]
    confidences = scores_to_confidence(scores, metric)
    best = torch.full((class_count,), -1.0, device=query.device)
    for index in range(class_count):
        mask = labels.to(query.device) == index
        if mask.any():
            best[index] = confidences[mask].max()
    return best.cpu()


def rank_classes(class_scores, classes, top_k, threshold):
    values = class_scores.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(-values)
    ranked = []
    for class_id in order[:top_k]:
        confidence = float(values[class_id])
        if confidence < threshold:
            continue
        ranked.append((int(class_id), classes[int(class_id)], confidence))
    return ranked
