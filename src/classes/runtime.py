from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.model import BoundingBox
from capsules.Fewshot.src.classes.encoder import (
    load_encoder,
    query_scores,
    scores_to_confidence,
)
from capsules.Fewshot.src.classes.gallery import (
    class_prototypes,
    collect_gallery,
    embed_paths,
    match_query,
    preprocess_bgr,
    rank_classes,
)
from capsules.Fewshot.src.models.PackageModel import Detection, PackageModel
from capsules.Fewshot.src.utils.response import (
    build_prototypical_response,
    build_siamese_response,
)
from capsules.Fewshot.src.utils.frames import load_frame
from capsules.Fewshot.src.utils.storage import (
    download_storage_file,
    extract_storage_zip,
    resolve_device,
)


def _config_param(config, name, default=None):
    value = Application().get_param(config, name)
    return default if value is None else value


def _executor_name(config):
    value = Application().get_param(config, "ConfigExecutor")
    if isinstance(value, dict):
        return value.get("name")
    return value


class FewShotRuntime(Capsule):
    matching = ""

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.input_detections = self.request.get_param("inputDetections")
        self.metric = self.request.get_param("ConfigDistanceMetric") or "euclidean"
        self.threshold = float(self.request.get_param("ConfigConfidenceThreshold") or 0.3)
        self.top_k = int(self.request.get_param("ConfigNumPredictions") or 1)
        self.imgsz = int(self.bootstrap.get("imgsz") or self.request.get_param("ConfigImgsz") or 224)
        self.encoder = self.bootstrap["encoder"]
        self.classes = self.bootstrap["classes"]
        self.references = self.bootstrap["references"]
        self.labels = self.bootstrap["labels"]
        self.matching = self.matching or self.bootstrap.get("matching") or "siamese"
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        logger = LoggerManager()
        matching = _executor_name(config) or "Siamese"
        matching = "prototypical" if matching == "Prototypical" else "siamese"
        device = resolve_device(_config_param(config, "ConfigDevice", "CPU"))
        imgsz = int(_config_param(config, "ConfigImgsz", 224) or 224)
        backbone = _config_param(config, "ConfigSiameseBackbone", "resnet18")
        embedding_dim = int(_config_param(config, "ConfigEmbeddingDim", 128) or 128)
        metric = _config_param(config, "ConfigDistanceMetric", "euclidean")

        weights_id = _config_param(config, "ConfigWeights")
        gallery_id = _config_param(config, "ConfigGallery")
        if weights_id is None:
            raise ValueError("Encoder weights are required.")
        if gallery_id is None:
            raise ValueError("A gallery zip is required.")

        weights_path, _ = download_storage_file(weights_id)
        encoder, meta = load_encoder(weights_path, backbone, embedding_dim, device)
        if meta.get("distance") and not _config_param(config, "ConfigDistanceMetric"):
            metric = meta["distance"]

        extract_dir = extract_storage_zip(gallery_id)
        samples, classes = collect_gallery(extract_dir)
        if not samples:
            raise ValueError("Gallery zip does not contain any images.")

        embeddings, labels = embed_paths(encoder, samples, imgsz)
        if matching == "prototypical":
            references = class_prototypes(embeddings, labels, len(classes))
            labels = None
        else:
            references = embeddings

        logger.info(
            f"Fewshot - {matching} gallery ready "
            f"({len(classes)} identities, {len(samples)} images, device={device}, imgsz={imgsz})"
        )
        return {
            "encoder": encoder,
            "classes": classes,
            "references": references,
            "labels": labels,
            "matching": matching,
            "metric": metric,
            "imgsz": imgsz,
            "device": device,
        }

    def _as_images(self):
        images = self.image
        if images is None:
            return []
        if isinstance(images, list):
            return images
        return [images]

    def _as_detections(self):
        detections = self.input_detections
        if detections is None:
            return []
        if isinstance(detections, list):
            return detections
        return [detections]

    def _detections_for_image(self, incoming, img_uid):
        if not incoming:
            return []
        if img_uid is None:
            return incoming
        keyed = []
        unkeyed = []
        for detection in incoming:
            det_uid = getattr(detection, "imgUID", None)
            if det_uid is None and isinstance(detection, dict):
                det_uid = detection.get("imgUID")
            if det_uid is None:
                unkeyed.append(detection)
            elif det_uid == img_uid:
                keyed.append(detection)
        return keyed or unkeyed

    def _detection_bbox(self, detection):
        box = getattr(detection, "boundingBox", None)
        if box is None and isinstance(detection, dict):
            box = detection.get("boundingBox")
        if box is None:
            return None
        if hasattr(box, "left"):
            return BoundingBox(
                left=float(box.left),
                top=float(box.top),
                width=float(box.width),
                height=float(box.height),
            )
        return BoundingBox(
            left=float(box.get("left", 0)),
            top=float(box.get("top", 0)),
            width=float(box.get("width", 0)),
            height=float(box.get("height", 0)),
        )

    def _crop(self, frame, bbox):
        height, width = frame.shape[:2]
        left = int(max(0, bbox.left))
        top = int(max(0, bbox.top))
        right = int(min(width, bbox.left + bbox.width))
        bottom = int(min(height, bbox.top + bbox.height))
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    def _predict_array(self, frame, img_uid, bbox):
        query = preprocess_bgr(frame, self.imgsz).unsqueeze(0)
        query = self.encoder.embed(query)
        if self.labels is None:
            scores = query_scores(query, self.references.to(query.device), self.metric)[0]
            class_scores = scores_to_confidence(scores, self.metric).cpu()
        else:
            class_scores = match_query(
                query, self.references, self.labels, len(self.classes), self.metric
            )
        for class_id, class_label, confidence in rank_classes(
            class_scores, self.classes, self.top_k, self.threshold
        ):
            self.detections.append(
                Detection(
                    boundingBox=bbox,
                    confidence=confidence,
                    classId=class_id,
                    classLabel=class_label,
                    imgUID=img_uid,
                )
            )

    def run(self):
        self.detections = []
        incoming = self._as_detections()
        for image in self._as_images():
            frame = load_frame(image, self.redis_db)
            array = frame.value
            img_uid = getattr(frame, "uID", None)
            if incoming:
                for detection in self._detections_for_image(incoming, img_uid):
                    bbox = self._detection_bbox(detection)
                    if bbox is None:
                        continue
                    crop = self._crop(array, bbox)
                    if crop is None:
                        continue
                    self._predict_array(crop, img_uid, bbox)
            else:
                height, width = array.shape[:2]
                bbox = BoundingBox(left=0, top=0, width=float(width), height=float(height))
                self._predict_array(array, img_uid, bbox)

        if self.matching == "prototypical":
            return build_prototypical_response(context=self)
        return build_siamese_response(context=self)
