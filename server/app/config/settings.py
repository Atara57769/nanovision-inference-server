"""Load and validate the model configuration (app/config/model_config.yaml).

The YAML is read once, at startup, into frozen typed dataclasses. This module is
the only place that knows the YAML key names; the rest of the server reads
attributes (``settings.batch_size``), never string keys. Every problem -- a
missing or misspelled key, a wrong type, an unknown model name -- stops the
server at startup with a message naming the key, instead of surfacing at the
first request.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from app.models.model_name import ModelName

# The server/ folder. Relative checkpoint paths start here, wherever the server is launched from.
SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("model_config.yaml")

# Optional environment variables, read once at startup, so a deployment can use
# its own files without editing the repository.
CONFIG_PATH_ENV = "NANOVISION_MODEL_CONFIG"  # replaces DEFAULT_CONFIG_PATH
CHECKPOINT_PATH_ENV = "NANOVISION_CHECKPOINT"  # replaces the active model's checkpoint

_TOP_LEVEL_KEYS = {"active_model", "device", "batch_size", "footer_height", "models"}
_MODEL_KEYS = {"checkpoint", "tile_size", "overlap"}


class SettingsError(ValueError):
    """The model configuration is missing, malformed or inconsistent."""


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"  # CUDA when available, otherwise CPU


@dataclass(frozen=True)
class ModelSettings:
    """What belongs to one model: its weights and the tiling it was trained for."""

    checkpoint: Path
    tile_size: int
    overlap: int


@dataclass(frozen=True)
class Settings:
    active_model: ModelName
    device: Device
    batch_size: int
    footer_height: int  # a property of the images, so shared by every model
    models: Mapping[ModelName, ModelSettings]

    @property
    def active(self) -> ModelSettings:
        """The settings of the model this server serves."""

        return self.models[self.active_model]


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Read the YAML file into validated ``Settings``; raise ``SettingsError`` naming the bad key."""

    with path.open(encoding="utf-8") as stream:
        text = stream.read()
    try:
        # safe_load builds only plain data (dicts, lists, numbers, strings), never objects.
        settings = _parse(yaml.safe_load(text))
        validate_settings(settings)
    except yaml.YAMLError as error:
        raise SettingsError(f"{path}: not valid YAML: {error}") from None
    except SettingsError as error:
        raise SettingsError(f"{path}: {error}") from None
    return settings


def load_settings_from_environment(environ: Mapping[str, str] = os.environ) -> Settings:
    """The settings of the running server: the default files, unless the environment names others."""

    # Relative paths in the environment start at server/ too: one rule for every path the server reads.
    config_path = environ.get(CONFIG_PATH_ENV)
    settings = load_settings(_path(config_path, CONFIG_PATH_ENV) if config_path else DEFAULT_CONFIG_PATH)
    checkpoint = environ.get(CHECKPOINT_PATH_ENV)
    if not checkpoint:
        return settings
    active = replace(settings.active, checkpoint=_path(checkpoint, CHECKPOINT_PATH_ENV))
    return replace(settings, models={**settings.models, settings.active_model: active})


def validate_settings(settings: Settings) -> None:
    """Refuse values that are well-formed but unusable, reporting every problem at once."""

    problems = []
    if settings.active_model not in settings.models:
        problems.append(f"active_model is '{settings.active_model}', but models has no entry for it")
    if settings.batch_size <= 0:
        problems.append(f"batch_size must be positive, got {settings.batch_size}")
    if settings.footer_height < 0:
        problems.append(f"footer_height must not be negative, got {settings.footer_height}")
    for name, model in settings.models.items():
        if model.tile_size <= 0:
            problems.append(f"models.{name}.tile_size must be positive, got {model.tile_size}")
        elif not 0 <= model.overlap < model.tile_size:
            # overlap == tile_size would mean a stride of 0: the tiles never advance.
            problems.append(
                f"models.{name}.overlap must be at least 0 and less than tile_size "
                f"({model.tile_size}), got {model.overlap}"
            )
    if problems:
        raise SettingsError("; ".join(problems))


# Parsing: raw YAML values -> typed fields. Each helper names the key it reads.


def _parse(raw: Any) -> Settings:
    top = _mapping(raw, "the top level", _TOP_LEVEL_KEYS)
    models = {}
    for raw_name, raw_model in _mapping(top["models"], "models").items():
        name = _model_name(raw_name, "models")  # first: a wrong name makes the rest meaningless
        where = f"models.{name}"
        entry = _mapping(raw_model, where, _MODEL_KEYS)
        models[name] = ModelSettings(
            checkpoint=_path(entry["checkpoint"], f"{where}.checkpoint"),
            tile_size=_whole_number(entry["tile_size"], f"{where}.tile_size"),
            overlap=_whole_number(entry["overlap"], f"{where}.overlap"),
        )
    return Settings(
        active_model=_model_name(top["active_model"], "active_model"),
        device=_device(top["device"], "device"),
        batch_size=_whole_number(top["batch_size"], "batch_size"),
        footer_height=_whole_number(top["footer_height"], "footer_height"),
        models=models,
    )


def _mapping(value: Any, where: str, keys: set[str] | None = None) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise SettingsError(f"{where} must be a mapping of keys to values, got {value!r}")
    if keys is not None:
        missing = keys - value.keys()
        # Rejecting unknown keys catches typos: a misspelled optional key would
        # otherwise be ignored without a word.
        unknown = value.keys() - keys
        if missing:
            raise SettingsError(f"{where} is missing: {', '.join(sorted(missing))}")
        if unknown:
            raise SettingsError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))} (a typo?)")
    return value


def _whole_number(value: Any, where: str) -> int:
    # bool is a subclass of int in Python, so `true` would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(f"{where} must be a whole number, got {value!r}")
    return value


def _model_name(value: Any, where: str) -> ModelName:
    try:
        return ModelName(value)
    except ValueError:
        raise SettingsError(f"{where}: unknown model {value!r}; known models: {', '.join(ModelName)}") from None


def _device(value: Any, where: str) -> Device:
    try:
        return Device(value)
    except ValueError:
        raise SettingsError(f"{where}: unknown device {value!r}; choose one of: {', '.join(Device)}") from None


def _path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"{where} must be a file path, got {value!r}")
    path = Path(value)
    return path if path.is_absolute() else SERVER_ROOT / path
