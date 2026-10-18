"""Measure connected particles in a segmentation mask, in pixel units."""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from app.models.classes import SegmentationClass
from app.analysis.statistics import summarize


def analyze_particles(mask: np.ndarray) -> dict:
    """Use 8-connected foreground; touching particles are one component."""
    foreground = mask != SegmentationClass.BACKGROUND
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3)))
    objects = ndimage.find_objects(labels)
    particles = []
    for label, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region = labels[bounds] == label
        classes = mask[bounds]
        area = int(region.sum())
        padded = np.pad(region, 1)
        perimeter = int(np.count_nonzero(padded[1:, :] != padded[:-1, :]) +
                        np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
        dark = region & (classes == SegmentationClass.DARK)
        bright = region & (classes == SegmentationClass.BRIGHT)
        interface = int(np.count_nonzero((dark[1:, :] & bright[:-1, :]) | (bright[1:, :] & dark[:-1, :])) +
                        np.count_nonzero((dark[:, 1:] & bright[:, :-1]) | (bright[:, 1:] & dark[:, :-1])))
        coords = np.argwhere(region).mean(axis=0) + [bounds[0].start, bounds[1].start]
        dark_area, bright_area = int(dark.sum()), int(bright.sum())
        particles.append({
            "id": label, "centroid": [float(coords[1]), float(coords[0])],
            "area": area, "perimeter": perimeter,
            "circularity": float(4 * np.pi * area / perimeter**2),
            "cu_area_fraction": dark_area / area,
            "ag_area_fraction": bright_area / area,
            "cu_ag_area_ratio": dark_area / bright_area if bright_area else None,
            "interface_length": interface,
            "nearest_particle_distance": None,
        })
    if len(particles) > 1:
        centers = np.array([p["centroid"] for p in particles])
        distances, _ = cKDTree(centers).query(centers, k=2)
        for particle, distance in zip(particles, distances[:, 1]):
            particle["nearest_particle_distance"] = float(distance)
    fields = ("area", "perimeter", "circularity", "cu_area_fraction",
              "ag_area_fraction", "cu_ag_area_ratio", "interface_length",
              "nearest_particle_distance")
    return {"particles": particles, "statistics": {
        field: summarize([p[field] for p in particles if p[field] is not None])
        for field in fields
    }, "units": {"length": "pixels", "area": "pixels_squared"}}
