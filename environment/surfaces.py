"""贴面区和避让面解析（input.json 里的 routing_surfaces / avoidance_surfaces）。"""
import json
import numpy as np


class RoutingSurface:
    """一个贴面区：点 + 走向方向。"""

    def __init__(self, part_id, direction, point, dir_vec):
        self.part_id = part_id
        self.direction = direction    # "X-"/"Y-"/"Z+"/"Y+"/"CUSTOM"
        self.point = np.asarray(point, float)
        self.dir = np.asarray(dir_vec, float)
        n = np.linalg.norm(self.dir)
        self.dir = self.dir / n if n > 1e-9 else self.dir

    def __repr__(self):
        return 'RoutingSurface(%s,%s,%s)' % (self.part_id, self.direction, np.round(self.point, 1))


class Surfaces:
    """贴面区（routing_surfaces）+ 局部避让面（avoidance_surfaces）。"""

    def __init__(self, request):
        self.routing_surfaces = []
        rd = request.get('routing_direction', {})
        for s in rd.get('routing_surfaces', []):
            self.routing_surfaces.append(RoutingSurface(
                s.get('part_id'), s.get('direction'), s.get('point_on_surface', s.get('point')), s.get('dir')))
        # 局部避让面（soft 约束，逐面给 avoidance_distance）
        self.avoidance_surfaces = request.get('avoidance_surfaces', []) or []

    @property
    def n_routing(self):
        return len(self.routing_surfaces)

    @property
    def n_avoidance(self):
        return len(self.avoidance_surfaces)


def load_surfaces(case_dir):
    with open(case_dir / 'input.json', encoding='utf-8-sig') as f:
        request = json.load(f)
    return Surfaces(request)
