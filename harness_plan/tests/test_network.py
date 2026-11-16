import numpy as np
import pytest

from harness_plan.smooth_layer import smooth_network
from harness_plan.result_layer import build_topology


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_whole_route_shared_curve_and_fixed_clips(device):
    positions = np.array([[0., 0., 0.], [10., 4., 0.], [20., 0., 0.],
                          [30., 8., 0.], [40., 12., 0.], [30., -8., 0.], [40., -12., 0.]])
    class Map:
        vs = 1.
        origin = np.zeros(3)
        def pos(self, node):
            return positions[node]
    paths = {'a': [0, 1, 2, 3, 4], 'b': [0, 1, 2, 5, 6]}
    geometry, offsets = smooth_network(Map(), paths, {0, 2, 4, 6}, {2: [1., 0., 0.]}, device)
    np.testing.assert_allclose(geometry['a'][:offsets['a'][2] + 1],
                               geometry['b'][:offsets['b'][2] + 1], atol=1e-10, rtol=0)
    for endpoint, path in paths.items():
        for occurrence, node in enumerate(path):
            if node in {0, 2, 4, 6}:
                np.testing.assert_array_equal(geometry[endpoint][offsets[endpoint][occurrence]], positions[node])


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_return_visits_use_explicit_sample_offsets(device):
    positions = np.array([[0., 0., 0.], [10., 0., 0.], [20., 3., 0.],
                          [20.123456789, 10., 0.], [30., 0., 0.]])
    class Map:
        vs = 1.
        origin = np.zeros(3)
        def pos(self, node):
            return positions[node]
    paths = {'a': [0, 1, 2, 3, 2, 1, 4], 'b': [0, 1, 4]}
    geometry, offsets = smooth_network(Map(), paths, set(range(5)), {}, device)
    case = dict(start=dict(pos=positions[0], dir=np.array([1., 0., 0.])),
                ends=[dict(id='a', pos=positions[4], dir=np.array([-1., 0., 0.])),
                      dict(id='b', pos=positions[4], dir=np.array([-1., 0., 0.]))],
                clips=[dict(id='clip', pos=positions[3], dir=np.array([0., 1., 0.]))],
                prefs={}, mode='one_click', part_category='test')
    metrics = {key: dict(length=float(np.linalg.norm(np.diff(value, axis=0), axis=1).sum()))
               for key, value in geometry.items()}
    topology, exported = build_topology(case, Map(), paths, [0, 4, 4, 3], {},
                                        {key: value.tolist() for key, value in geometry.items()}, metrics, [],
                                        path_samples=offsets)
    np.testing.assert_array_equal(topology['clip_points'][0]['position_mm'], positions[3])
    for harness in topology['harnesses']:
        pieces = []
        for first, last, sid in zip(harness['cp_sequence'], harness['cp_sequence'][1:], harness['segment_sequence']):
            segment = topology['segments'][sid]
            piece = np.array(topology['edge_paths'][str(sid)]['path_mm'])
            if first != segment['from_cp_id']:
                piece = piece[::-1]
            pieces.append(piece if not pieces else piece[1:])
        np.testing.assert_allclose(np.vstack(pieces), exported[harness['id']], atol=1e-8, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_endpoint_direction_is_directed(device):
    points = np.array([[0., 0., 0.], [10., 0., 0.]])
    class Map:
        def pos(self, node):
            return points[node]
    geometry, _ = smooth_network(Map(), {'end': [0, 1]}, {0, 1},
                                 {0: [-1., 0., 0.], 1: [-1., 0., 0.]}, device, step=0.01)
    path = geometry['end']
    assert path[1, 0] < path[0, 0]
    assert path[-1, 0] > path[-2, 0]
