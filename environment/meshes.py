"""三类 STL 加载：障碍(env_) / 贴面(sheet_part) / 避让(avoid_part)。"""
import numpy as np
import trimesh


def classify_stl(name):
    lower = name.lower()
    if lower.startswith('sheet_part') or lower.startswith('sheet_combined'):
        return 'sheet'
    if lower.startswith('avoid_part') or lower.startswith('avoidance_combined'):
        return 'avoid'
    return 'env'


class Meshes:
    """环境网格：障碍（硬避障）、贴面（软约束）、避让（软约束）。"""

    def __init__(self, input_dir):
        stls = sorted(input_dir.rglob('*.stl')) if input_dir.is_dir() else []
        self.files = {
            'env': [f for f in stls if classify_stl(f.name) == 'env'],
            'sheet': [f for f in stls if classify_stl(f.name) == 'sheet'],
            'avoid': [f for f in stls if classify_stl(f.name) == 'avoid'],
        }
        self.meshes = {k: self._load(v) for k, v in self.files.items()}

    def _load(self, files):
        if not files:
            return None
        parts = [trimesh.load_mesh(f, process=False) for f in files]
        for p in parts:
            p.merge_vertices()
        return trimesh.util.concatenate(parts)

    def _distance(self, key, points):
        mesh = self.meshes.get(key)
        if mesh is None:
            return np.full(len(points), float('inf'))
        return trimesh.proximity.closest_point(mesh, np.asarray(points, float))[1]

    # 障碍：硬避障（线束表面到障碍 >= d）
    def obstacle_distance(self, points):
        return self._distance('env', points)

    # 贴面：软约束（越小越贴面）
    def sheet_distance(self, points):
        return self._distance('sheet', points)

    # 避让：软约束（到局部避让面距离）
    def avoid_distance(self, points):
        return self._distance('avoid', points)

    def summary(self):
        return {k: len(v) for k, v in self.files.items()}

    def has(self, key):
        return self.meshes.get(key) is not None


def load_meshes(case_dir):
    return Meshes(case_dir / 'input')
