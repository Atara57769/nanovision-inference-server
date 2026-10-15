"""The segmentation classes: what each value in a returned mask means.

Defined once, without torch, so every layer (including the API) can import it.
The order is the research label order (NanoVision code/amat-2/src/utils.py
``CLASS_NAMES`` @ d8f35ed) and also the order of the model's output channels:
output channel k is ``SegmentationClass(k)``. ``BaseModel.load()`` refuses a
checkpoint trained with a different order, so the two cannot drift apart.

Research masks also use 255 for "ignore"; that is an annotation value, not a
class, and the server never returns it.
"""

from enum import IntEnum


class SegmentationClass(IntEnum):
    BACKGROUND = 0  # substrate
    DARK = 1  # dark part of a particle
    BRIGHT = 2  # bright part of a particle


NUM_CLASSES = len(SegmentationClass)

# Lower-case names, the way the research training config stores them. Sorted by
# value, so position k always names output channel k, even if someone reorders
# or renumbers the members above.
CLASS_NAMES = tuple(member.name.lower() for member in sorted(SegmentationClass))
