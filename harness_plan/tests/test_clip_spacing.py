import json
import numpy as np
import pytest

from route import route
from harness_plan.topology_layer import select_spaced_clips


def selection(positions, edges, candidate_nodes, minimum=60.):
    positions = np.asarray(positions, float)
    case = dict(input_request=dict(candidate_clips=[dict(clip_id=str(i)) for i in candidate_nodes]),
                clips=[dict(id=str(i), pos=positions[i], dir=np.array([1., 0., 0.])) for i in candidate_nodes])
    topology = dict(control_points=[dict(id=i) for i in range(len(positions))],
                    segments=[dict(id=i, from_cp_id=a, to_cp_id=b) for i, (a, b) in enumerate(edges)],
                    edge_paths={str(i): dict(path_mm=positions[[a, b]].tolist()) for i, (a, b) in enumerate(edges)},
                    clip_points=[dict(control_point_id=i) for i in candidate_nodes])
    return select_spaced_clips(case, topology, [True] * len(candidate_nodes), minimum)


@pytest.mark.parametrize('distance,excluded_count', [(0., 1), (59.999, 1), (60., 0), (60.001, 0)])
def test_strict_network_distance_threshold(distance, excluded_count):
    excluded, nearest = selection([[0, 0, 0], [distance, 0, 0]], [(0, 1)], [0, 1])
    assert len(excluded) == excluded_count
    assert nearest == distance


def test_distance_crosses_branch_point():
    excluded, nearest = selection([[0, 0, 0], [-20, 0, 0], [0, 20, 0], [50, 0, 0]],
                                  [(0, 1), (0, 2), (0, 3)], [1, 2])
    assert nearest == 40
    assert len(excluded) == 1


def test_euclidean_neighbours_can_both_remain_after_detour():
    excluded, nearest = selection([[0, 0, 0], [0, 100, 0], [5, 0, 0]], [(0, 1), (1, 2)], [0, 2])
    assert nearest > 200
    assert excluded == []


def test_keep_two_outer_clips_instead_of_one_middle_clip():
    excluded, _ = selection([[0, 0, 0], [40, 0, 0], [80, 0, 0]], [(0, 1), (1, 2)], [0, 1, 2])
    assert [item['clip_id'] for item in excluded] == ['1']


def test_disconnected_clips_are_not_spacing_conflicts():
    excluded, nearest = selection([[0, 0, 0], [1, 0, 0]], [], [0, 1])
    assert excluded == []
    assert nearest is None


def test_route_replans_and_records_selection(tmp_path):
    case = tmp_path / 'case'
    case.mkdir()
    request = dict(endpoints=dict(starts=[dict(id='start', pos=[0, 0, 0], dir=[1, 0, 0])],
                                  ends=[dict(id='end', pos=[120, 0, 0], dir=[-1, 0, 0])]),
                   candidate_clips=[dict(clip_id='a', pos=[30, 0, 0], dir=[1, 0, 0]),
                                    dict(clip_id='b', pos=[50, 0, 0], dir=[1, 0, 0])],
                   control_points=[dict(point_id='control', pos=[100, 0, 0], dir=[1, 0, 0], order=1)])
    (case / 'input.json').write_text(json.dumps(request), encoding='utf8')
    result = route(case, voxel=15)
    selected = result['topology']['metadata']['clip_selection']
    assert selected['input_count'] == 2
    assert selected['selected_count'] == 1
    assert len(selected['excluded_clips']) == 1
    assert selected['excluded_clips'][0]['conflicts'][0]['distance_mm'] < 60
    assert result['statistics']['n_clip_points'] == 2  # selected customer clip + mandatory control point
    assert result['statistics']['reached_endpoints'] == 1
    assert result['success']
