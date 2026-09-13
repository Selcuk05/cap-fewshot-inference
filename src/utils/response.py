from sdks.novavision.src.helper.package import PackageHelper
from capsules.Fewshot.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    FewShotOutputs,
    OutputDetections,
    Siamese,
    SiameseResponse,
    Prototypical,
    PrototypicalResponse,
)


def _build(context, executor_cls, response_cls):
    outputs = FewShotOutputs(
        outputDetections=OutputDetections(value=context.detections)
    )
    response = response_cls(outputs=outputs)
    executor = executor_cls(value=response)
    package_configs = PackageConfigs(executor=ConfigExecutor(value=executor))
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    return package.build_model(context)


def build_siamese_response(context):
    return _build(context, Siamese, SiameseResponse)


def build_prototypical_response(context):
    return _build(context, Prototypical, PrototypicalResponse)
