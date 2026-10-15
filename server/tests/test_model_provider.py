"""ModelProvider and the model registry: the configured model is created and loaded once."""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.config.settings import Device, ModelSettings, Settings
from app.inference.inference_pipeline import InferencePipeline
from app.model_management.model_provider import ModelLoadError, ModelProvider, resolve_device
from app.model_management.model_registry import MODEL_TYPES, create_model
from app.models.model_name import ModelName
from app.models.unetpp_model import UNetPlusPlusModel
from tests.helpers import NO_NORMALISATION, FakeModel, StripeNetwork, research_unetpp


def make_settings(
    checkpoint: Path, *, tile_size: int = 256, batch_size: int = 8, footer_height: int = 0
) -> Settings:
    return Settings(
        active_model=ModelName.UNETPP,
        device=Device.CPU,
        batch_size=batch_size,
        footer_height=footer_height,
        models={ModelName.UNETPP: ModelSettings(checkpoint=checkpoint, tile_size=tile_size, overlap=128)},
    )


def fake_factory(name: ModelName, **model_settings: int) -> FakeModel:
    """Stands in for create_model, so these tests do not depend on U-Net++."""

    return FakeModel(**model_settings)


@pytest.fixture
def fake_checkpoint(write_checkpoint) -> Path:
    return write_checkpoint(StripeNetwork().state_dict(), model_name=FakeModel.name, **NO_NORMALISATION)


# --- Registry ----------------------------------------------------------------


def test_every_model_name_has_a_registered_class() -> None:
    assert set(MODEL_TYPES) == set(ModelName)


def test_registry_creates_a_model_that_is_not_loaded_yet() -> None:
    model = create_model(ModelName.UNETPP, tile_size=256, overlap=128, batch_size=8, footer_height=0)

    assert isinstance(model, UNetPlusPlusModel)
    assert not model.is_loaded


# --- Provider ----------------------------------------------------------------


def test_provider_loads_the_configured_unetpp(write_checkpoint) -> None:
    checkpoint = write_checkpoint(research_unetpp().state_dict(), model_name="unetpp")
    provider = ModelProvider(make_settings(checkpoint))

    model = provider.load_active_model()

    assert isinstance(model, UNetPlusPlusModel)
    assert model.is_loaded
    assert provider.get_model() is model
    assert model.info().name == "unetpp"


def test_the_model_is_loaded_only_once(fake_checkpoint: Path) -> None:
    provider = ModelProvider(make_settings(fake_checkpoint), model_factory=fake_factory)
    model = provider.load_active_model()

    with pytest.raises(RuntimeError, match="already loaded"):
        provider.load_active_model()
    assert provider.get_model() is provider.get_model() is model
    assert model.build_calls == 1


def test_get_model_before_startup_fails_clearly(fake_checkpoint: Path) -> None:
    provider = ModelProvider(make_settings(fake_checkpoint), model_factory=fake_factory)

    with pytest.raises(RuntimeError, match="no model is loaded yet"):
        provider.get_model()


def test_the_provider_passes_the_configuration_to_the_model(fake_checkpoint: Path) -> None:
    settings = make_settings(fake_checkpoint, batch_size=4, footer_height=50)
    model = ModelProvider(settings, model_factory=fake_factory).load_active_model()

    prepared = model.preprocess(np.zeros((768, 1024, 3), dtype=np.uint8))

    assert prepared.metadata.tiling.original_shape == (718, 1024)  # footer_height applied
    assert [len(batch) for batch in prepared.batches] == [4] * 8 + [3]  # batch_size applied


def test_any_registered_model_runs_through_the_pipeline(fake_checkpoint: Path) -> None:
    # The provider and the pipeline only know BaseModel, so another model
    # class needs no change in either: switching models is configuration only.
    model = ModelProvider(make_settings(fake_checkpoint), model_factory=fake_factory).load_active_model()

    result = InferencePipeline(model).run(np.zeros((300, 300, 3), dtype=np.uint8))

    assert result.mask.shape == (300, 300)
    assert result.model_name == "fake-stripes"


def test_a_missing_checkpoint_names_the_config_key(tmp_path: Path) -> None:
    provider = ModelProvider(make_settings(tmp_path / "missing.pt"), model_factory=fake_factory)

    with pytest.raises(ModelLoadError, match=r"models\.unetpp\.checkpoint"):
        provider.load_active_model()
    with pytest.raises(RuntimeError, match="no model is loaded yet"):
        provider.get_model()  # nothing half-loaded is handed out


def test_a_config_that_disagrees_with_the_checkpoint_is_refused(fake_checkpoint: Path) -> None:
    # The checkpoint was trained on 256px crops; the config asks for 512px tiles.
    provider = ModelProvider(make_settings(fake_checkpoint, tile_size=512), model_factory=fake_factory)

    with pytest.raises(ModelLoadError, match="256px crops"):
        provider.load_active_model()


# --- Device ------------------------------------------------------------------


def test_cpu_device() -> None:
    assert resolve_device(Device.CPU) == torch.device("cpu")


def test_auto_device_follows_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device(Device.AUTO).type == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has a CUDA GPU")
def test_cuda_without_a_gpu_fails_at_startup() -> None:
    with pytest.raises(ModelLoadError, match="CUDA is not available"):
        resolve_device(Device.CUDA)
