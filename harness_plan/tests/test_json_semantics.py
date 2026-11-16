import numpy as np

from harness_plan.result_layer import build_topology


def test_clip_tangents_and_distinct_harness_counts():
    positions = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    class Map:
        vs = 1.
        origin = np.zeros(3)
        def pos(self, node):
            return positions[node]
    case = dict(start=dict(pos=positions[0], dir=np.array([1., 0., 0.])),
                ends=[dict(id='end', pos=positions[2], dir=np.array([-1., 0., 0.]))],
                clips=[dict(id='clip', pos=positions[1], dir=np.array([-1., 0., 0.]))],
                prefs=dict(clearance_mm=6.), mode='one_click', part_category='test')
    path = [0, 1, 2, 1, 2]
    topology, _ = build_topology(case, Map(), {'end': path}, [0, 2, 1], {0: -1, 1: 0, 2: 1},
                                 {'end': positions[path].tolist()}, {'end': dict(length=40.)}, [],
                                 path_samples={'end': np.arange(len(path))})
    assert topology['clip_points'][0]['tangent'] == [1., 0., 0.]
    for segment in topology['segments']:
        assert segment['n_harnesses_passing'] == 1
        assert segment['safe_dist_mm'] == segment['radius_mm'] + 6.
        clip_id = topology['clip_points'][0]['control_point_id']
        if segment['from_cp_id'] == clip_id:
            assert segment['from_tangent'] == topology['clip_points'][0]['tangent']
        if segment['to_cp_id'] == clip_id:
            assert segment['to_tangent'] == topology['clip_points'][0]['tangent']
