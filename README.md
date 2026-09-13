# Fewshot

> Runs Siamese and Prototypical encoders trained by **cap-trainer** and returns identity matches as NovaVision `Detection` objects.

**Category:** Capsule

**Status:** Beta

**Last Updated:** 13.09.2026

---

## 1. Overview

Fewshot is the matching capsule for encoders produced by Trainer's Few-Shot task. Trainer only trains a shared embedding encoder. This package loads that encoder, embeds a gallery of identities, embeds each query, and scores the query against the gallery.

Pick **Siamese** or **Prototypical**. Both use the same encoder weights. Siamese scores the query against every gallery embedding and keeps the best match per identity. Prototypical replaces each identity with the mean of its gallery embeddings, then scores against those prototypes.

Use this package after Trainer when the set of identities is small or changes often, and a closed-set classifier is not a good fit. Downstream draw / filter / if packages consume the `Detection` list the same way they consume YOLO classify output.

**Typical use cases:**

- Identify a person, part, or product against a folder of enrollment images
- Re-identify crops from an upstream detector (face, person, or object) without retraining a classifier
- Swap the gallery zip to add or remove identities without training a new encoder

---

## 2. Inputs

Both executors share the same inputs.

| Field Name | Kind/Type | Required | Default | Description |
|---|---|---|---|---|
| `inputImage` | Image or list[Image] | Yes | — | Query frame. When `inputDetections` is empty, the whole image is embedded. |
| `inputDetections` | Detection or list[Detection] | No | — | Optional boxes from an upstream detector. Each box is cropped and matched; the original box is kept on the output. |

> Derived from the package `Inputs` / `Input` classes. A field that accepts a single object or a list is shown explicitly.

---

## 3. Configuration Parameters

Shared parameters appear on both Siamese and Prototypical. `ConfigExecutor` is omitted here.

| Parameter | Type | Field Type (UI control) | Allowed Values / Range | Default | Description |
|---|---|---|---|---|---|
| `ConfigWeights` | number | filePicker | `.pt`, `.onnx`, `.torchscript` | — | Encoder artifact from Trainer. Prefer `best.pt`. The `.pt` checkpoint usually stores `backbone`, `embedding_dim`, and `distance`. |
| `ConfigGallery` | number | filePicker | `.zip` | — | Identity-folder zip used as the support / gallery set. Same layout as Trainer few-shot data. |
| `ConfigSiameseBackbone` | object | dropdownlist | resnet18, mobilenetV2 | resnet18 | Encoder trunk used when the checkpoint has no metadata. Ignored when the `.pt` already stores `backbone`. |
| `ConfigEmbeddingDim` | number | textInput | `[32, 512]` | `128` | Embedding size used when the checkpoint has no metadata. Must match the trained encoder. Ignored for ONNX / TorchScript and for `.pt` files that already store `embedding_dim`. |
| `ConfigDistanceMetric` | object | dropdownlist | euclidean, cosine | euclidean | Distance in embedding space. Use the same metric the encoder was trained with. Euclidean is the Prototypical default. |
| `ConfigDevice` | object | dropdownlist | CPU, GPU | — | Inference device. GPU uses `cuda:0` when CUDA is available; otherwise the run falls back to CPU. |
| `ConfigImgsz` | number | textInput | `[32, 1280]` | `224` | Square image size fed to the encoder. Must match the size used at training time. |
| `ConfigConfidenceThreshold` | number | textInput | `[0.0, 1.0]` | `0.3` | Minimum match confidence. Cosine uses clamped similarity; Euclidean uses `1 / (1 + distance)`. |
| `ConfigNumPredictions` | number | textInput | `[1, 20]` | `1` | Number of top identity matches to emit per image or crop, each as a `Detection`. |

> Derived from the package `Configs` classes, including validation bounds and dropdown option labels where present. `ConfigExecutor` is not included.

---

## 4. Outputs

| Field Name | Kind/Type | Description |
|---|---|---|
| `outputDetections` | list[Detection] | Identity matches. Each item has `classLabel` (gallery folder name), `classId`, `confidence`, `boundingBox`, and `imgUID`. Whole-image matches use a box that covers the frame. Crop matches reuse the incoming box. |

> Derived from the package `Outputs` classes.

---

## 5. Use Case Examples

**Use case 1: Whole-image identity match**

Trainer Few-Shot produced `best.pt` from `train/<identity>/*.jpg`. Point **Weights** at that file and **Gallery** at the same (or an updated) identity zip. Leave `inputDetections` unconnected. Each query frame becomes one or more `Detection` objects with the matched identity. Draw Label or Filter can consume the list.

**Use case 2: Detect, then identify**

A detector (YOLO, face, or similar) feeds `inputImage` and `inputDetections`. This capsule crops each box, embeds the crop, and writes the identity onto that same box. Use this when the query is a scene with several objects, not a single centered subject.

**Use case 3: New identities without retraining**

The encoder stays the same. Replace the gallery zip with a folder that adds or drops identities, then restart the executor (gallery has `restart: True`). Bootstrap re-embeds the new support set. No Trainer job is required unless the embedding space itself needs to change.

---

## 6. Limitations and Notes

### Gallery format

The gallery is a **single `.zip` from storage**. A zip that contains exactly one top-level folder is unwrapped. Image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

Accepted layouts:

```
train/<identity>/*.jpg
val/<identity>/*.jpg      # optional; included in the gallery when present
```

or class / identity folders at the zip root (no `train/`). All images in those folders are used. Unlike Trainer, nothing is held out.

- At least one identity with at least one image is required.
- Identity names on the output are the folder names.
- Unsafe zip paths (escaping the extract dir) are rejected.

### Weights

- Trainer `best.pt` is the primary format. It usually includes `state_dict`, `backbone`, `embedding_dim`, `distance`, `architecture`, and `classes`. Gallery folder names still define the live class list; checkpoint `classes` are not used as the gallery.
- ONNX and TorchScript exports from Trainer are accepted. Backbone and embedding dim are not needed for those files.
- A raw state dict (no metadata) needs **Encoder** and **Embedding Dim** to match the training run.

### Matching behavior

- **Siamese** keeps every gallery embedding and takes the best score per identity.
- **Prototypical** averages all gallery embeddings of an identity into one prototype, then scores the query against those means.
- Confidence is not a closed-set softmax. Cosine is clamped similarity in `[0, 1]`. Euclidean is `1 / (1 + distance)`. Raise **Confidence Threshold** to reject unknown queries.
- **Image Size** must match training. Trainer few-shot typically uses `224`, not the YOLO default `640`.
- **Distance** should match training. A cosine-trained encoder scored with Euclidean (or the reverse) will rank poorly.

### Runtime

- **Bootstrap** downloads weights, extracts the gallery, and embeds every gallery image. Changing weights, gallery, backbone, embedding dim, device, image size, or task restarts the executor.
- First CUDA step can sit quietly for a while (compile / cuDNN). That is not a hang if the gallery-ready log already appeared.
- GPU falls back to CPU when CUDA is not available.
- ONNX needs `onnxruntime` (and `onnxruntime-gpu` if you select GPU).
- This capsule does not train. Produce weights with **cap-trainer** Few-Shot first.
