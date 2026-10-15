"""Create and load the configured model once, at startup, and hand it out afterwards.

One ``ModelProvider`` lives as long as the server process. The startup code
(the lifespan in app/main.py) would use it like this:

    provider = ModelProvider(load_settings())
    model = provider.load_active_model()
    pipeline = InferencePipeline(model)

Nothing on the request path loads weights; requests only use the loaded model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from app.config.settings import Device, Settings
from app.model_management.checkpoint_download import ensure_checkpoint
from app.model_management.model_registry import create_model
from app.models.base_model import BaseModel

# Same signature as create_model; tests pass one that builds a stand-in model.
ModelFactory = Callable[..., BaseModel]


class ModelLoadError(RuntimeError):
    """The configured model could not be created or loaded.

    One exception type for every cause, so startup code can decide in one place
    whether to stop the server or start it as "not ready".
    """


def resolve_device(choice: Device) -> torch.device:
    """Turn the configured device into a torch device, refusing CUDA where there is none."""

    if choice is Device.AUTO:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice is Device.CUDA and not torch.cuda.is_available():
        raise ModelLoadError(
            "device is 'cuda', but CUDA is not available here (no GPU, or a CPU-only torch build)"
        )
    return torch.device(choice)


class ModelProvider:
    """Owns the server's single model instance."""

    def __init__(self, settings: Settings, model_factory: ModelFactory = create_model) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._model: BaseModel | None = None

    def load_active_model(self) -> BaseModel:
        """Create the configured model and load its weights. Runs once; a second call is a bug."""

        if self._model is not None:
            raise RuntimeError("the model is already loaded; load_active_model() runs once, at startup")
        name = self._settings.active_model
        model_settings = self._settings.active
        device = resolve_device(self._settings.device)
        try:
            checkpoint = ensure_checkpoint(
                model_settings.checkpoint,
                model_settings.checkpoint_url,
                model_settings.checkpoint_sha256,
            )
            model = self._model_factory(
                name,
                tile_size=model_settings.tile_size,
                overlap=model_settings.overlap,
                batch_size=self._settings.batch_size,
                footer_height=self._settings.footer_height,
            )
            model.load(checkpoint, device)
        except Exception as error:
            # Whatever the cause (missing file, corrupt file, incompatible
            # checkpoint), the configured model is unusable. Name the config key
            # to fix and keep the original error chained underneath.
            raise ModelLoadError(
                f"could not load the '{name}' model from {model_settings.checkpoint} "
                f"(set in models.{name}.checkpoint): {error}"
            ) from error
        self._model = model
        return model

    def get_model(self) -> BaseModel:
        """The loaded model; fails if startup has not loaded it yet."""

        if self._model is None:
            raise RuntimeError("no model is loaded yet; load_active_model() must run at startup first")
        return self._model
