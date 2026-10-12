"""Names of the models this server can serve.

One closed list, shared by the YAML config, the model registry and each model
class, so a typo in the config fails at startup instead of at the first request.
Each value equals ``model.name`` in the research training config, so it can be
compared directly with what a checkpoint says it contains.
"""

from enum import StrEnum


class ModelName(StrEnum):
    UNETPP = "unetpp"
