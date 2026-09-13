from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Image,
    Detection,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class Detection(Detection):
    imgUID: Optional[str] = None


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], Detection]
    type: str = "object"

    class Config:
        title = "Detections"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


def _file_picker(short_description, extensions):
    return {
        "shortDescription": short_description,
        "class": "portalium\\storage\\widgets\\FilePicker",
        "options": {
            "multiple": 0,
            "returnAttribute": ["name"],
            "name": "app::logo_wide",
            "fileExtensions": extensions,
        },
    }


class ConfigWeights(Config):
    """
    Encoder artifact from Few-Shot training. Accepts the Trainer checkpoint
    (.pt) or a portable ONNX / TorchScript export of the same encoder.
    """

    name: Literal["ConfigWeights"] = "ConfigWeights"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"
    restart: Literal[True] = True

    class Config:
        json_schema_extra = _file_picker("Encoder Weights", ["pt", "onnx", "torchscript"])
        title = "Weights"


class ConfigGallery(Config):
    """
    Identity-folder zip used as the support / gallery set. Same layout as
    Trainer few-shot data: train/<identity>/*.jpg or class folders at the root.
    """

    name: Literal["ConfigGallery"] = "ConfigGallery"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"
    restart: Literal[True] = True

    class Config:
        json_schema_extra = _file_picker("Gallery Zip", ["zip"])
        title = "Gallery"


class OptionResNet18(Config):
    name: Literal["resnet18"] = "resnet18"
    value: Literal["resnet18"] = "resnet18"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "ResNet-18"


class OptionMobilenetV2(Config):
    name: Literal["mobilenetV2"] = "mobilenetV2"
    value: Literal["mobilenetV2"] = "mobilenetV2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "MobileNetV2"


class ConfigSiameseBackbone(Config):
    """
    Encoder trunk used when the checkpoint has no metadata (raw state dict).
    Ignored when the .pt file already stores backbone and embedding dim.
    """

    name: Literal["ConfigSiameseBackbone"] = "ConfigSiameseBackbone"
    value: Union[OptionResNet18, OptionMobilenetV2]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Encoder"
        json_schema_extra = {"shortDescription": "Few-Shot Encoder"}


class ConfigEmbeddingDim(Config):
    """
    Embedding size used when the checkpoint has no metadata. Must match the
    trained encoder. Ignored for ONNX / TorchScript and for .pt files that
    already store embedding_dim.
    """

    name: Literal["ConfigEmbeddingDim"] = "ConfigEmbeddingDim"
    value: int = Field(ge=32, le=512, default=128)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[32, 512]"] = "[32, 512]"
    restart: Literal[True] = True

    class Config:
        title = "Embedding Dim"
        json_schema_extra = {"shortDescription": "Embedding Dimension"}


class OptionEuclidean(Config):
    name: Literal["euclidean"] = "euclidean"
    value: Literal["euclidean"] = "euclidean"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Euclidean"


class OptionCosine(Config):
    name: Literal["cosine"] = "cosine"
    value: Literal["cosine"] = "cosine"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Cosine"


class ConfigDistanceMetric(Config):
    """
    Distance in embedding space. Use the same metric the encoder was trained with.
    Euclidean is the Prototypical default; cosine is scale-invariant.
    """

    name: Literal["ConfigDistanceMetric"] = "ConfigDistanceMetric"
    value: Union[OptionEuclidean, OptionCosine]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Distance"
        json_schema_extra = {"shortDescription": "Embedding Distance"}


class OptionDeviceCPU(Config):
    name: Literal["cpu"] = "cpu"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class OptionDeviceGPU(Config):
    name: Literal["gpu"] = "gpu"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDevice(Config):
    """
    Device used for embedding and matching. GPU uses CUDA when available and
    falls back to CPU otherwise.
    """

    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[OptionDeviceCPU, OptionDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"
        json_schema_extra = {"shortDescription": "Inference Device"}


class ConfigImgsz(Config):
    """
    Square image size fed to the encoder. Must match the size used at training
    time. Few-shot runs typically use 224.
    """

    name: Literal["ConfigImgsz"] = "ConfigImgsz"
    value: int = Field(ge=32, le=1280, default=224)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[32, 1280]"] = "[32, 1280]"
    restart: Literal[True] = True

    class Config:
        title = "Image Size"
        json_schema_extra = {"shortDescription": "Encoder Image Size"}


class ConfigConfidenceThreshold(Config):
    """
    Minimum match confidence for a detection to be returned. Scores come from
    cosine similarity or a 1 / (1 + distance) map of Euclidean distance.
    """

    name: Literal["ConfigConfidenceThreshold"] = "ConfigConfidenceThreshold"
    value: float = Field(ge=0.0, le=1.0, default=0.3)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Confidence Threshold"
        json_schema_extra = {"shortDescription": "Minimum Match Confidence"}


class ConfigNumPredictions(Config):
    """
    Number of top identity matches to emit per image, each as a Detection.
    """

    name: Literal["ConfigNumPredictions"] = "ConfigNumPredictions"
    value: int = Field(ge=1, le=20, default=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1, 20]"] = "[1, 20]"

    class Config:
        title = "Top Predictions"
        json_schema_extra = {"shortDescription": "Top-K Identities"}


class FewShotInputs(Inputs):
    inputImage: InputImage
    inputDetections: Optional[InputDetections] = None


class FewShotConfigs(Configs):
    configWeights: ConfigWeights
    configGallery: ConfigGallery
    configSiameseBackbone: ConfigSiameseBackbone
    configEmbeddingDim: ConfigEmbeddingDim
    configDistanceMetric: ConfigDistanceMetric
    configDevice: ConfigDevice
    configImgsz: ConfigImgsz
    configConfidenceThreshold: ConfigConfidenceThreshold
    configNumPredictions: ConfigNumPredictions


class FewShotOutputs(Outputs):
    outputDetections: OutputDetections


class SiameseRequest(Request):
    inputs: Optional[FewShotInputs]
    configs: FewShotConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SiameseResponse(Response):
    outputs: FewShotOutputs


class Siamese(Config):
    name: Literal["Siamese"] = "Siamese"
    value: Union[SiameseRequest, SiameseResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Siamese"
        json_schema_extra = {"target": {"value": 0}}


class PrototypicalRequest(Request):
    inputs: Optional[FewShotInputs]
    configs: FewShotConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class PrototypicalResponse(Response):
    outputs: FewShotOutputs


class Prototypical(Config):
    name: Literal["Prototypical"] = "Prototypical"
    value: Union[PrototypicalRequest, PrototypicalResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Prototypical"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    """
    Matching method. Siamese scores the query against every gallery embedding.
    Prototypical scores against the mean embedding of each identity.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[Siamese, Prototypical]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Select Matching Task"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Fewshot"] = "Fewshot"
