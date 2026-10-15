"""Test doubles and builders shared by several test files."""

import segmentation_models_pytorch as smp
import torch
from torch import nn

from app.models.base_model import BaseModel

DARK_BELOW, BRIGHT_ABOVE = 0.25, 0.75
# StripeNetwork reads gray values directly, so its checkpoints use an identity normalisation.
NO_NORMALISATION = {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)}


class StripeNetwork(nn.Module):
    """Pointwise stand-in: gray < 0.25 -> DARK, gray > 0.75 -> BRIGHT, otherwise BACKGROUND.

    Pointwise means each pixel's class depends only on that pixel, so the correct
    mask is known in advance for any image, and any mistake in tiling, ordering,
    averaging or cropping shows up as a wrong pixel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sharpness = nn.Parameter(torch.tensor(100.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x[:, :1]
        background = torch.zeros_like(gray)
        dark = (DARK_BELOW - gray) * self.sharpness
        bright = (gray - BRIGHT_ABOVE) * self.sharpness
        return torch.cat([background, dark, bright], dim=1)  # channel k = SegmentationClass(k)


class FakeModel(BaseModel):
    name = "fake-stripes"

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.build_calls = 0
        self.network: StripeNetwork | None = None

    def _build_network(self) -> nn.Module:
        self.build_calls += 1
        self.network = StripeNetwork()
        return self.network


def research_unetpp() -> nn.Module:
    """U-Net++ built independently of the server code, exactly as research src/models/unetpp.py does."""

    return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
