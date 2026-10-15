"""The BaseModel contract, tested with a tiny pointwise stand-in network (no real checkpoint)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from app.inference.inference_pipeline import InferencePipeline
from app.inference.tiling import split_to_tiles
from app.models import base_model
from app.models.base_model import BaseModel, ReconstructionMetadata
from app.models.classes import SegmentationClass
from tests.helpers import BRIGHT_ABOVE, DARK_BELOW, NO_NORMALISATION, FakeModel, StripeNetwork

TILE, OVERLAP = 256, 128
CPU = torch.device("cpu")


def expected_mask(gray_uint8: np.ndarray) -> np.ndarray:
    """What StripeNetwork must produce, computed without any model code."""

    gray = gray_uint8 / 255.0
    mask = np.full(gray.shape, SegmentationClass.BACKGROUND, dtype=np.uint8)
    mask[gray < DARK_BELOW] = SegmentationClass.DARK
    mask[gray > BRIGHT_ABOVE] = SegmentationClass.BRIGHT
    return mask


def as_rgb(gray_uint8: np.ndarray) -> np.ndarray:
    return np.repeat(gray_uint8[:, :, None], 3, axis=2)


@pytest.fixture
def make_model(write_checkpoint):
    def make(*, batch_size: int = 8, footer_height: int = 0, **checkpoint_fields) -> FakeModel:
        model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=batch_size, footer_height=footer_height)
        fields = {**NO_NORMALISATION, **checkpoint_fields}
        path = write_checkpoint(StripeNetwork().state_dict(), model_name=FakeModel.name, **fields)
        model.load(path, CPU)
        return model

    return make


# --- Interface ---------------------------------------------------------------


def test_base_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)  # type: ignore[abstract]


def test_runs_unchanged_through_the_inference_pipeline(make_model) -> None:
    model = make_model()
    gray = np.random.default_rng(0).integers(0, 256, (300, 400), dtype=np.uint8)

    result = InferencePipeline(model).run(as_rgb(gray))

    assert result.mask.shape == (300, 400)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1, 2}
    assert result.model_name == "fake-stripes"
    assert result.model_version.startswith("v1-step100-")


# --- Request path ------------------------------------------------------------


def test_class_ids_follow_the_final_class_order(make_model) -> None:
    model = make_model()
    # Three vertical stripes: black, mid-gray, white.
    gray = np.zeros((256, 300), dtype=np.uint8)
    gray[:, 100:200] = 128
    gray[:, 200:] = 255

    mask = InferencePipeline(model).run(as_rgb(gray)).mask

    assert np.all(mask[:, :100] == SegmentationClass.DARK) and SegmentationClass.DARK == 1
    assert np.all(mask[:, 100:200] == SegmentationClass.BACKGROUND) and SegmentationClass.BACKGROUND == 0
    assert np.all(mask[:, 200:] == SegmentationClass.BRIGHT) and SegmentationClass.BRIGHT == 2


@pytest.mark.parametrize("shape", [(768, 1024), (700, 900), (100, 50)])
def test_tiled_mask_equals_the_known_answer(make_model, shape: tuple[int, int]) -> None:
    # Covers padding (100x50), uneven overlap (700x900) and the real SEM size.
    model = make_model()
    gray = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)

    mask = InferencePipeline(model).run(as_rgb(gray)).mask

    np.testing.assert_array_equal(mask, expected_mask(gray))


def test_full_sem_image_becomes_35_tiles_in_batches(make_model) -> None:
    model = make_model(batch_size=8)

    prepared = model.preprocess(np.zeros((768, 1024, 3), dtype=np.uint8))

    assert [len(batch) for batch in prepared.batches] == [8, 8, 8, 8, 3]
    assert all(batch.shape[1:] == (3, TILE, TILE) for batch in prepared.batches)
    assert all(batch.dtype == torch.float32 for batch in prepared.batches)
    assert prepared.metadata.image_shape == (768, 1024)


def test_preprocess_applies_the_checkpoints_normalisation(make_model) -> None:
    # Different values per channel, to prove each channel uses its own mean/std.
    mean, std = (0.1, 0.2, 0.3), (0.5, 0.25, 0.125)
    model = make_model(mean=mean, std=std)

    prepared = model.preprocess(np.full((256, 256, 3), 51, dtype=np.uint8))  # 51 / 255 = 0.2

    tile = prepared.batches[0][0]
    for channel in range(3):
        expected = (0.2 - mean[channel]) / std[channel]
        np.testing.assert_allclose(tile[channel].numpy(), expected, atol=1e-6)


@pytest.mark.parametrize(
    "image",
    [
        np.full((256, 256), 51, dtype=np.uint8),  # gray
        np.full((256, 256, 3), 51, dtype=np.uint8),  # gray stored as RGB
        np.full((256, 256, 4), 51, dtype=np.uint8),  # gray stored as RGBA
        np.full((256, 256, 1), 51, dtype=np.uint8),  # gray with a channel axis
        np.full((256, 256), 13107, dtype=np.uint16),  # 16-bit: 13107 / 65535 = 0.2
    ],
)
def test_supported_formats_give_identical_input(make_model, image: np.ndarray) -> None:
    model = make_model()
    reference = model.preprocess(np.full((256, 256), 51, dtype=np.uint8)).batches[0]

    np.testing.assert_array_equal(model.preprocess(image).batches[0].numpy(), reference.numpy())


def test_colour_is_converted_like_training_did() -> None:
    # PIL's convert("L") (the training loader) weights red by 0.299: 255 * 0.299 -> 76.
    pure_red = np.zeros((4, 4, 3), dtype=np.uint8)
    pure_red[:, :, 0] = 255

    np.testing.assert_array_equal(base_model.to_gray01(pure_red), np.float32(76 / 255))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((256, 256), dtype=np.float32),  # unknown value scale
        np.zeros((256, 256), dtype=np.int32),
        np.zeros((256, 256, 2), dtype=np.uint8),  # two channels
        np.zeros((256, 256, 3), dtype=np.uint16),  # 16-bit colour
        np.zeros((0, 256, 3), dtype=np.uint8),  # empty
        np.zeros((2, 256, 256, 3), dtype=np.uint8),  # a batch, not one image
    ],
)
def test_unsupported_images_are_rejected(make_model, image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        make_model().preprocess(image)


def test_probabilities_are_averaged_before_argmax() -> None:
    # Two tiles share columns 128-255. Averaging probabilities gives DARK there;
    # averaging logits would give BRIGHT, because tile B's extreme -100 drags
    # DARK's mean logit down. Research averages probabilities.
    model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)
    _, tiling = split_to_tiles(np.zeros((3, 256, 384), dtype=np.float32), TILE, OVERLAP)
    metadata = ReconstructionMetadata(image_shape=(256, 384), tiling=tiling)
    tile_a = torch.tensor([0.0, 10.0, 0.0]).reshape(3, 1, 1).expand(3, TILE, TILE)  # sure it is DARK
    tile_b = torch.tensor([0.0, -100.0, 1.0]).reshape(3, 1, 1).expand(3, TILE, TILE)  # leans BRIGHT

    mask = model.postprocess([torch.stack([tile_a, tile_b])], metadata)

    assert np.all(mask[:, :128] == SegmentationClass.DARK)  # tile A only
    assert np.all(mask[:, 128:256] == SegmentationClass.DARK)  # overlap: probability average
    assert np.all(mask[:, 256:] == SegmentationClass.BRIGHT)  # tile B only


def test_footer_rows_are_background_and_never_predicted(make_model) -> None:
    model = make_model(footer_height=50)
    white = np.full((300, 300, 3), 255, dtype=np.uint8)  # white would be BRIGHT if predicted

    prepared = model.preprocess(white)
    mask = InferencePipeline(model).run(white).mask

    assert prepared.metadata.tiling.original_shape == (250, 300)  # footer not tiled
    assert mask.shape == (300, 300)
    assert np.all(mask[:250] == SegmentationClass.BRIGHT)
    assert np.all(mask[250:] == SegmentationClass.BACKGROUND)


def test_footer_taller_than_the_image_is_rejected(make_model) -> None:
    with pytest.raises(ValueError, match="footer_height"):
        make_model(footer_height=50).preprocess(np.zeros((40, 300, 3), dtype=np.uint8))


# --- Loading -----------------------------------------------------------------


def test_load_puts_trained_weights_in_an_inference_ready_network(write_checkpoint) -> None:
    trained = StripeNetwork()
    trained.sharpness.data.fill_(42.0)
    model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)

    model.load(write_checkpoint(trained.state_dict(), model_name=FakeModel.name), CPU)

    assert model.is_loaded
    assert model.network.sharpness.item() == 42.0
    assert model.network.training is False  # eval() mode
    assert not any(parameter.requires_grad for parameter in model.network.parameters())


def test_version_identifies_the_exact_checkpoint_file(write_checkpoint) -> None:
    path = write_checkpoint(StripeNetwork().state_dict(), model_name=FakeModel.name, step=2500)
    model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)
    model.load(path, CPU)

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert model.info().version == f"v1-step2500-{file_hash[:8]}"


@pytest.mark.parametrize(
    ("checkpoint_fields", "reason"),
    [
        ({"model_name": "unetpp"}, "holds a 'unetpp' model"),
        ({"classes": ("background", "bright", "dark")}, "trained with classes"),  # the old order
        ({"crop_size": 512}, "512px crops"),
        ({"mean": (0.5,), "std": (0.5,)}, "does not have 3 channels"),
    ],
)
def test_incompatible_checkpoints_are_refused(write_checkpoint, checkpoint_fields: dict, reason: str) -> None:
    fields = {"model_name": FakeModel.name, **checkpoint_fields}
    path = write_checkpoint(StripeNetwork().state_dict(), **fields)
    model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)

    with pytest.raises(ValueError, match=reason):
        model.load(path, CPU)
    assert not model.is_loaded  # a failed load leaves nothing half-done


def test_weights_that_do_not_fit_are_refused(write_checkpoint) -> None:
    path = write_checkpoint({"some_other_layer": torch.zeros(1)}, model_name=FakeModel.name)

    with pytest.raises(ValueError, match="do not fit"):
        FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8).load(path, CPU)


def test_a_bare_state_dict_is_not_a_training_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "weights_only.pt"
    torch.save(StripeNetwork().state_dict(), path)

    with pytest.raises(ValueError, match="not a NanoVision training checkpoint"):
        FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8).load(path, CPU)


def test_missing_checkpoint_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8).load(tmp_path / "missing.pt", CPU)


def test_using_the_model_before_load_fails_clearly() -> None:
    model = FakeModel(tile_size=TILE, overlap=OVERLAP, batch_size=8)

    with pytest.raises(RuntimeError, match="not loaded"):
        model.preprocess(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="not loaded"):
        model.predict(torch.zeros(1, 3, TILE, TILE))
    with pytest.raises(RuntimeError, match="not loaded"):
        model.info()


def test_predictions_never_reload_the_checkpoint(make_model, monkeypatch) -> None:
    calls = []
    real_load_checkpoint = base_model.load_checkpoint

    def counting_load_checkpoint(path: Path):
        calls.append(path)
        return real_load_checkpoint(path)

    monkeypatch.setattr(base_model, "load_checkpoint", counting_load_checkpoint)
    model = make_model()
    pipeline = InferencePipeline(model)

    for _ in range(3):
        pipeline.run(np.zeros((300, 300, 3), dtype=np.uint8))

    assert len(calls) == 1
    assert model.build_calls == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"tile_size": 0, "overlap": 0, "batch_size": 8},
        {"tile_size": TILE, "overlap": TILE, "batch_size": 8},  # stride would be 0
        {"tile_size": TILE, "overlap": -1, "batch_size": 8},
        {"tile_size": TILE, "overlap": OVERLAP, "batch_size": 0},
        {"tile_size": TILE, "overlap": OVERLAP, "batch_size": 8, "footer_height": -1},
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict) -> None:
    with pytest.raises(ValueError):
        FakeModel(**settings)
