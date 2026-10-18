import numpy as np
import pytest

from app.analysis.particle_analyzer import analyze_particles


def test_measurements_for_two_particles():
    mask = np.array([[1, 2, 0, 1], [1, 2, 0, 1]], dtype=np.uint8)
    result = analyze_particles(mask)
    first, second = result['particles']
    assert first['area'] == 4
    assert first['perimeter'] == 8
    assert first['circularity'] == pytest.approx(np.pi / 4)
    assert first['cu_area_fraction'] == 0.5
    assert first['ag_area_fraction'] == 0.5
    assert first['cu_ag_area_ratio'] == 1
    assert first['interface_length'] == 2
    assert first['nearest_particle_distance'] == pytest.approx(2.5)
    assert second['cu_ag_area_ratio'] is None
    assert result['statistics']['area']['mean'] == 3
    assert result['statistics']['area']['variance'] == 1
    assert sum(result['statistics']['area']['histogram']['counts']) == 2
    assert result['statistics']['cu_ag_area_ratio']['count'] == 1


def test_empty_and_diagonally_touching_particles():
    empty = analyze_particles(np.zeros((2, 2), dtype=np.uint8))
    assert empty['particles'] == []
    assert empty['statistics']['area']['mean'] is None
    touching = analyze_particles(np.array([[1, 0], [0, 2]], dtype=np.uint8))
    assert len(touching['particles']) == 1
    assert touching['particles'][0]['nearest_particle_distance'] is None
    assert touching['particles'][0]['interface_length'] == 0
