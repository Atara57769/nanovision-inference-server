"""Which class implements each model name: the one list the server chooses from.

Adding a model means adding its name to ``ModelName`` and its class here; a test
fails if the two lists disagree.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.models.base_model import BaseModel
from app.models.model_name import ModelName
from app.models.unetpp_model import UNetPlusPlusModel

MODEL_TYPES: Mapping[ModelName, type[BaseModel]] = {
    ModelName.UNETPP: UNetPlusPlusModel,
}


def create_model(
    name: ModelName, *, tile_size: int, overlap: int, batch_size: int, footer_height: int
) -> BaseModel:
    """A new, not yet loaded instance of the model registered under ``name``."""

    model_type = MODEL_TYPES.get(name)
    if model_type is None:
        raise ValueError(f"no model class is registered for '{name}'; registered: {', '.join(MODEL_TYPES)}")
    return model_type(tile_size=tile_size, overlap=overlap, batch_size=batch_size, footer_height=footer_height)
