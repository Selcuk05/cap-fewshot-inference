from sdks.novavision.src.base.model import Image as ImageModel
from sdks.novavision.src.media.image import Image


def _as_dict(image):
    if image is None:
        return None
    if isinstance(image, dict):
        return dict(image)
    if hasattr(image, "model_dump"):
        return image.model_dump()
    if hasattr(image, "dict"):
        return image.dict()
    payload = {}
    for key in ("name", "uID", "mimeType", "encoding", "value", "r_key", "shape_key", "type"):
        if hasattr(image, key):
            payload[key] = getattr(image, key)
    return payload or None


def load_frame(image, redis_db):
    if image is None:
        raise ValueError("inputImage is missing.")
    if isinstance(image, list):
        if not image:
            raise ValueError("inputImage list is empty.")
        image = image[0]

    value = getattr(image, "value", None)
    if hasattr(value, "shape"):
        return image

    payload = _as_dict(image)
    if not payload:
        raise ValueError("inputImage is missing or is not an image object.")

    r_key = payload.get("r_key") or ""
    if r_key:
        if redis_db is None:
            raise RuntimeError("Redis is not available on this executor.")
        payload["value"] = redis_db.redis_get_frame(r_key)
        shape_key = payload.get("shape_key") or ""
        if shape_key:
            payload["shape_key"] = redis_db.redis_get_frame(shape_key)
        payload["r_key"] = ""

    encoding = payload.get("encoding")
    if encoding == "base64":
        content, shape_key = Image.decode64(payload["value"], payload["shape_key"])
    elif encoding == "bytes":
        content, shape_key = Image.decode_bytes(payload["value"], payload["shape_key"])
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    mime_type = payload.get("mimeType") or "image/jpg"
    if mime_type == "image/jpeg":
        mime_type = "image/jpg"

    frame = ImageModel(
        name=payload.get("name") or "inputImage",
        uID=payload.get("uID") or "",
        mimeType=mime_type,
        encoding=encoding,
        value=content,
        shape_key=shape_key,
        r_key=payload.get("r_key") or "",
        type=payload.get("type") or "Image",
    )
    for key, item in payload.items():
        if key not in {
            "name",
            "uID",
            "mimeType",
            "encoding",
            "value",
            "r_key",
            "shape_key",
            "type",
        }:
            setattr(frame, key, item)
    return frame
