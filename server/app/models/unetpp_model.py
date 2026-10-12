"""U-Net++ with a ResNet34 encoder: the team's selected model.

Built exactly like NanoVision code/amat-2/src/models/unetpp.py ``build`` @ d8f35ed
(segmentation-models-pytorch ``UnetPlusPlus``, ResNet34, 3 input channels,
3 classes), except that it never asks for ImageNet weights: the trained weights
come from the checkpoint, and downloading at every startup would be slow and
would need network access.
"""

import segmentation_models_pytorch as smp
from torch import nn

from app.models.base_model import INPUT_CHANNELS, BaseModel
from app.models.classes import NUM_CLASSES
from app.models.model_name import ModelName

ENCODER = "resnet34"


class UNetPlusPlusModel(BaseModel):
    name = ModelName.UNETPP

    def _build_network(self) -> nn.Module:
        return smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=INPUT_CHANNELS,
            classes=NUM_CLASSES,
        )
