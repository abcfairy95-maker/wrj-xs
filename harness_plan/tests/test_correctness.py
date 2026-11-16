import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from route import route
from harness_plan.data_layer import Collision
from harness_plan.map_layer import Roadmap
from harness_plan.pathfind_layer import backtrack, wavefront_batch
from harness_plan.smooth_layer import dedupe, hermite_smooth, smooth_gpu
from harness_plan.result_layer import build_topology


def test_gpu_preserves_endpoints_clips_and_junctions():
    points = np.array([[0., 0., 0.], [10., 10., 0.], [20., 5., 0.],
                       [30., 10., 0.], [40., 0., 0.]])
    result = smooth_gpu(points, [2], [[1., 0., 0.]], 'cpu', steps=5, fixed_indices=[1])
    np.testing.assert_array_equal(result[[0, 1, 2, 4]], points[[0, 1, 2, 4]])


def test_dedupe_and_sampling_preserve_close_anchors():
    points = np.array([[0., 0., 0.], [.2, 0., 0.], [2., 0., 0.]])
    kept = dedupe(points, fixed=[True, True, True])
    np.testing.assert_array_equal(kept, points)
    result = hermite_smooth(kept, np.tile([1., 0., 0.], (3, 1)))
    assert any(np.array_equal(point, points[1]) for point in result)


def test_wavefront_unreachable_and_path_direction():
    free = np.ones((5, 1, 1), bool)
    field = wavefront_batch(free, [(0, 0, 0)], 'cpu').numpy()[0]
    path = backtrack(field, (0, 0, 0), (4, 0, 0), free)
    assert path == [(i, 0, 0) for i in range(5)]
    free[2] = False
    field = wavefront_batch(free, [(0, 0, 0)], 'cpu').numpy()[0]
    assert np.isinf(field[4, 0, 0])
    assert backtrack(field, (0, 0, 0), (4, 0, 0), free) is None


def test_wavefront_reaches_long_detour():
    free = np.ones((7, 7, 1), bool)
    free[1, :6] = False
    free[3, 1:] = False
    free[5, :6] = False
    field = wavefront_batch(free, [(0, 0, 0)], 'cpu').numpy()[0]
    assert field[6, 0, 0] > sum(free.shape)
    assert backtrack(field, (0, 0, 0), (6, 0, 0), free) is not None


def test_gpu_map_does_not_connect_disconnected_voxels():
    roadmap = Roadmap(Collision([]), np.zeros(3), np.array([4., 0., 0.]), 1, 0, device='gpu')
    roadmap.idx[2] = -1
    start = int(roadmap.idx[0, 0, 0])
    end = int(roadmap.idx[4, 0, 0])
    assert np.isinf(roadmap.pair([start, end])[0, 1])
    assert roadmap.path(start, end) is None


def make_case(tmp_path, clips=()):
    case = tmp_path / 'case'
    case.mkdir()
    request = dict(endpoints=dict(starts=[dict(id='start', pos=[0, 0, 0], dir=[1, 0, 0])],
                                  ends=[dict(id='end', pos=[60, 0, 0], dir=[-1, 0, 0])]),
                   required_clips=list(clips), part_category='test')
    (case / 'input.json').write_text(json.dumps(request), encoding='utf-8')
    return case


def test_legacy_positional_arguments_and_default_output(tmp_path):
    case = make_case(tmp_path)
    result = route(case, 15)
    assert result['topology']['metadata']['dispatch']['part_category'] == 'test'
    assert result['success']
    assert result['topology']['metadata']['voxel_size_mm'] == 15
    assert (tmp_path / 'case_result_new.json').exists()
    assert '_device' not in result and '_runtime' not in result


def test_final_endpoint_geometry_controls_status(tmp_path, monkeypatch):
    import route as entry
    original = entry.smooth_network
    def shifted(*args, **kwargs):
        geometry, offsets = original(*args, **kwargs)
        for points in geometry.values():
            points[-1] += [10, 0, 0]
        return geometry, offsets
    monkeypatch.setattr(entry, 'smooth_network', shifted)
    result = route(make_case(tmp_path), 15)
    assert not result['success']
    assert result['statistics']['reached_endpoints'] == 0
    assert result['unreached_endpoints'][0]['reason'] == 'endpoint_geometry_mismatch'


def test_missing_clip_is_reported(tmp_path, monkeypatch):
    import route as entry
    original = entry.smooth_network
    def shifted(*args, **kwargs):
        geometry, offsets = original(*args, **kwargs)
        for points in geometry.values():
            points[1:-1] += [0, 100, 0]
        return geometry, offsets
    monkeypatch.setattr(entry, 'smooth_network', shifted)
    clip = dict(clip_id='clip', pos=[30, 20, 0], dir=[1, 0, 0])
    result = route(make_case(tmp_path, [clip]), 15)
    assert not result['success']
    assert result['topology']['metadata']['clip_coverage'] == dict(used=0, total=1)
    assert result['topology']['clip_constraint_summary']['violation_count'] == 1
    assert result['issues'][0]['code'] == 'CLIP_UNREACHED'
    assert result['topology']['clip_points'][0]['constraint_status'] == 'fail'


@pytest.mark.parametrize('nodes_on_path', [[0, 1, 2, 1, 3], [0, 1, 2, 1, 2, 3]])
def test_export_retains_return_traversal_and_segment_direction(nodes_on_path):
    positions = np.array([[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [20., 0., 0.]])
    class Map:
        vs = 1.
        origin = np.zeros(3)
        def pos(self, index):
            return positions[index]
    case = dict(start=dict(pos=positions[0], dir=np.array([1., 0., 0.])),
                ends=[dict(id='end', pos=positions[3], dir=np.array([-1., 0., 0.]))],
                clips=[dict(id='clip', pos=positions[2], dir=np.array([0., 1., 0.]))],
                prefs={}, mode='one_click', part_category='test')
    paths = {'end': nodes_on_path}
    geometry = {'end': positions[paths['end']].tolist()}
    topology, _ = build_topology(case, Map(), paths, [0, 3, 2],
                                 {0: -1, 1: 0, 2: 1, 3: 1}, geometry,
                                 {'end': dict(length=40.)}, [])
    harness = topology['harnesses'][0]
    assert len(harness['segment_sequence']) == len(nodes_on_path) - 1
    assert harness['segment_sequence'][1] == harness['segment_sequence'][2]
    controls = {point['id']: point['position_mm'] for point in topology['control_points']}
    for segment in topology['segments']:
        path = topology['edge_paths'][str(segment['id'])]['path_mm']
        np.testing.assert_array_equal(path[0], controls[segment['from_cp_id']])
        np.testing.assert_array_equal(path[-1], controls[segment['to_cp_id']])
