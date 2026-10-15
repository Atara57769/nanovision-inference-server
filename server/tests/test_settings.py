"""Loading and validating model_config.yaml."""

import copy
from pathlib import Path

import pytest
import yaml

from app.config.settings import (
    CHECKPOINT_PATH_ENV,
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG_PATH,
    SERVER_ROOT,
    Device,
    ModelSettings,
    Settings,
    SettingsError,
    load_settings,
    load_settings_from_environment,
    validate_settings,
)
from app.models.model_name import ModelName

VALID_CONFIG = {
    "active_model": "unetpp",
    "device": "cpu",
    "batch_size": 8,
    "footer_height": 0,
    "models": {"unetpp": {"checkpoint": "checkpoints/unetpp/best.pt", "tile_size": 256, "overlap": 128}},
}


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "model_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_the_shipped_config_is_valid() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.active_model is ModelName.UNETPP
    assert settings.active.tile_size == 256
    assert settings.active.overlap == 128
    assert settings.device is Device.CPU
    assert settings.footer_height == 50  # team decision: crop the info bar, as research did


def test_without_environment_variables_the_project_defaults_are_used() -> None:
    settings = load_settings_from_environment(environ={})

    assert settings == load_settings(DEFAULT_CONFIG_PATH)
    assert settings.active.checkpoint == SERVER_ROOT / "checkpoints" / "unetpp" / "best.pt"


def test_environment_variables_choose_the_config_and_the_checkpoint(tmp_path: Path) -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["batch_size"] = 2
    environ = {
        CONFIG_PATH_ENV: str(write_config(tmp_path, config)),
        CHECKPOINT_PATH_ENV: str(tmp_path / "deployed.pt"),
    }

    settings = load_settings_from_environment(environ)

    assert settings.batch_size == 2  # read from the other config file
    assert settings.active.checkpoint == tmp_path / "deployed.pt"


def test_relative_paths_in_the_environment_also_start_at_the_server_folder() -> None:
    settings = load_settings_from_environment({CHECKPOINT_PATH_ENV: "elsewhere/best.pt"})

    assert settings.active.checkpoint == SERVER_ROOT / "elsewhere" / "best.pt"


def test_yaml_becomes_typed_settings(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, VALID_CONFIG))

    assert settings == Settings(
        active_model=ModelName.UNETPP,
        device=Device.CPU,
        batch_size=8,
        footer_height=0,
        models={
            ModelName.UNETPP: ModelSettings(
                checkpoint=SERVER_ROOT / "checkpoints" / "unetpp" / "best.pt", tile_size=256, overlap=128
            )
        },
    )


def test_relative_checkpoint_paths_start_at_the_server_folder(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, VALID_CONFIG))

    assert settings.active.checkpoint.is_relative_to(SERVER_ROOT)
    assert (SERVER_ROOT / "app" / "config" / "settings.py").is_file()  # SERVER_ROOT really is server/


def test_absolute_checkpoint_paths_are_kept(tmp_path: Path) -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["models"]["unetpp"]["checkpoint"] = str(tmp_path / "best.pt")

    assert load_settings(write_config(tmp_path, config)).active.checkpoint == tmp_path / "best.pt"


def _set(key: str, value):
    return lambda config: config.update({key: value})


def _set_model(key: str, value):
    return lambda config: config["models"]["unetpp"].update({key: value})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(_set("active_model", "unet"), "unknown model 'unet'", id="unknown active model"),
        pytest.param(_set("models", {"unet": {}}), "unknown model 'unet'", id="unknown model under models"),
        pytest.param(_set("models", {}), "no entry for it", id="active model not configured"),
        pytest.param(_set("device", "gpu"), "unknown device 'gpu'", id="unknown device"),
        pytest.param(_set("batch_size", 0), "batch_size must be positive", id="zero batch size"),
        pytest.param(_set("batch_size", "8"), "batch_size must be a whole number", id="number as text"),
        pytest.param(_set("footer_height", True), "footer_height must be a whole number", id="boolean as number"),
        pytest.param(_set("footer_height", -1), "footer_height must not be negative", id="negative footer"),
        pytest.param(_set_model("tile_size", 0), "tile_size must be positive", id="zero tile size"),
        pytest.param(_set_model("overlap", 256), "overlap must be at least 0", id="overlap equals tile"),
        pytest.param(_set_model("checkpoint", ""), "checkpoint must be a file path", id="empty checkpoint"),
        pytest.param(lambda config: config.pop("batch_size"), "missing: batch_size", id="missing key"),
        pytest.param(_set("batch_sise", 8), "unknown keys: batch_sise", id="misspelled key"),
        pytest.param(_set_model("tile", 256), "unknown keys: tile", id="misspelled model key"),
    ],
)
def test_invalid_configs_stop_the_server(tmp_path: Path, change, message: str) -> None:
    config = copy.deepcopy(VALID_CONFIG)
    change(config)

    with pytest.raises(SettingsError, match=message):
        load_settings(write_config(tmp_path, config))


def test_errors_name_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "model_config.yaml"
    path.write_text("", encoding="utf-8")  # an empty file is not a mapping

    with pytest.raises(SettingsError, match="model_config.yaml: the top level must be a mapping"):
        load_settings(path)


def test_broken_yaml_is_reported_as_a_settings_error(tmp_path: Path) -> None:
    path = tmp_path / "model_config.yaml"
    path.write_text("active_model: [unetpp", encoding="utf-8")

    with pytest.raises(SettingsError, match="not valid YAML"):
        load_settings(path)


def test_every_problem_is_reported_at_once() -> None:
    settings = Settings(
        active_model=ModelName.UNETPP,
        device=Device.CPU,
        batch_size=0,
        footer_height=-1,
        models={ModelName.UNETPP: ModelSettings(checkpoint=Path("best.pt"), tile_size=256, overlap=128)},
    )

    with pytest.raises(SettingsError) as error:
        validate_settings(settings)

    assert "batch_size" in str(error.value) and "footer_height" in str(error.value)
