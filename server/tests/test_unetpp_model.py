"""U-Net++ with untrained (random) weights: architecture, shapes and strict loading.

Accuracy is not tested here -- that needs the real checkpoint and is checked
against the research predictor separately. These tests prove that a checkpoint
of the research architecture loads into the server model and runs end to end.
"""

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch

from app.inference.inference_pipeline import InferencePipeline
from app.models.classes import NUM_CLASSES
from app.models.model_name import ModelName
from app.models.unetpp_model import UNetPlusPlusModel
from tests.helpers import research_unetpp

CPU = torch.device("cpu")


@pytest.fixture
def unetpp(write_checkpoint) -> UNetPlusPlusModel:
    path = write_checkpoint(research_unetpp().state_dict(), model_name="unetpp")
    model = UNetPlusPlusModel(tile_size=256, overlap=128, batch_size=8)
    model.load(path, CPU)
    return model


def test_name_matches_the_research_config() -> None:
    assert UNetPlusPlusModel.name == ModelName.UNETPP == "unetpp"


def test_predict_returns_logits_at_tile_resolution(unetpp: UNetPlusPlusModel) -> None:
    with torch.inference_mode():
        logits = unetpp.predict(torch.zeros(2, 3, 256, 256))

    assert logits.shape == (2, NUM_CLASSES, 256, 256)


def test_whole_image_runs_through_the_pipeline(unetpp: UNetPlusPlusModel) -> None:
    # 200x300 needs both padding (height < 256) and two overlapping tiles (width).
    image = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)

    result = InferencePipeline(unetpp).run(image)

    assert result.mask.shape == (200, 300)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1, 2}
    assert result.model_name == "unetpp"
    assert result.model_version.startswith("v1-step100-")


def test_weights_of_another_architecture_are_refused(write_checkpoint) -> None:
    # Same encoder, different decoder: strict loading must notice.
    plain_unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    path = write_checkpoint(plain_unet.state_dict(), model_name="unetpp")

    with pytest.raises(ValueError, match="do not fit"):
        UNetPlusPlusModel(tile_size=256, overlap=128, batch_size=8).load(path, CPU)
