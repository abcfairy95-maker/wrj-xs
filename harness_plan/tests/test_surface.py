import sys
from pathlib import Path
sys.path.insert(0, r'C:\Users\Administrator\wrj-xs')
import numpy as np
import trimesh
from scipy.sparse.csgraph import dijkstra
from harness_plan.data_layer import SurfacePreference
from harness_plan.map_layer import Roadmap
from harness_plan.pathfind_layer import wavefront_batch, backtrack


def test_weighted_wavefront_matches_graph():
    class Empty:
        def dist(self, p):
            return np.full(len(p), np.inf)
    rm = Roadmap(Empty(), np.zeros(3), np.array([4, 4, 0]), 1, 0)
    rm.pref = np.zeros(rm.n)
    rm.pref[rm.idx[1:4, 2, 0]] = 100
    rm.grid_edges()
    source, target = (0, 2, 0), (4, 2, 0)
    costs = rm._voxel_cost()
    distances = wavefront_batch(rm.idx >= 0, [source], 'cpu', cost=costs).numpy()[0]
    exact = dijkstra(rm.graph(), indices=int(rm.idx[source]))
    np.testing.assert_allclose(distances[rm.idx >= 0], exact, atol=1e-5)
    path = backtrack(distances, source, target, rm.idx >= 0, cost=costs)
    assert sum((costs[a] + costs[b]) / 2 for a, b in zip(path, path[1:])) == exact[rm.idx[target]]
    assert len(path) > 5


def test_single_finite_surface_not_anchor_polyline(tmp_path):
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]])
    file = tmp_path / 'sheet.stl'
    mesh.export(file)
    surface = SurfacePreference([dict(path=str(file), direction=[0, 0, 1])], 12, 5)
    distances = surface.dist([[50, 50, 12], [50, 50, -12], [200, 50, 12]])
    assert distances[0] < 5
    assert distances[1] >= 24
    assert distances[2] >= 100
