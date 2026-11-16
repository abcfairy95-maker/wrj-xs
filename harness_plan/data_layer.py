"""数据层：输入解析、参数、碰撞模型（CPU/GPU 共享，不重复）。"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


# ================= 数据加载 =================
def load_case(case, input_file=None):
    case = Path(case)
    cand = []
    if input_file:
        cand += [case / input_file, case / 'input' / input_file]
    cand += [case / 'input.json', case / 'input' / 'input.json']
    jf = next(c for c in cand if c.is_file())
    req = json.load(open(jf, encoding='utf-8-sig'))
    ep = req.get('endpoints', {})
    stls = sorted((case / 'input').rglob('*.stl'))
    rd = req.get('routing_direction', {}).get('routing_surfaces', [])
    surf = [np.asarray(s.get('point_on_surface', s.get('point')), float) for s in rd]
    clips = [dict(id=c['clip_id'], pos=np.asarray(c['pos'], float), dir=unit(c['dir']))
             for c in req.get('required_clips', []) or req.get('candidate_clips', [])]
    # control_point 模式的控制点作为必经途经点（按 order 排序）
    for cp in sorted(req.get('control_points', []), key=lambda x: x.get('order', 0)):
        clips.append(dict(id=cp.get('point_id', cp.get('id')), pos=np.asarray(cp['pos'], float),
                          dir=unit(cp.get('dir', [1, 0, 0]))))
    return dict(
        input_request=req, task_id=case.name,
        start=dict(id=ep['starts'][0]['id'], pos=np.asarray(ep['starts'][0]['pos'], float),
                   dir=unit(ep['starts'][0]['dir'])),
        ends=[dict(id=e['id'], pos=np.asarray(e['pos'], float), dir=unit(e['dir']))
              for e in ep.get('ends', [])],
        clips=clips,
        prefs=req.get('preferences', {}),
        env=[str(f) for f in stls if not f.name.lower().startswith(('sheet_', 'avoid_'))],
        routing_surfaces=surf,
        surface_parts=[dict(path=str(case / 'input' / 'surface' / 'parts' /
                                     ('sheet_part_' + str(entry['part_id']) + '.stl')),
                            direction=entry.get('dir', [0, 0, 1])) for entry in rd],
        part_category=req.get('part_category', 'other'),
        mode=req.get('mode', 'one_click'),
    )


def params(prefs):
    D = float(prefs.get('max_harness_diameter_mm', 12))
    db = float(prefs.get('min_harness_diameter_mm', 6))
    r = float(prefs.get('bend_radius_factor', 2))
    return dict(D_t=D, d_b=db, r=r, clearance=D / 2 + float(prefs.get('clearance_mm', 6)),
                R_min=D * r)


# ================= 碰撞模型（障碍顶点 KDTree） =================
class Collision:
    def __init__(self, files, max_verts=500000):
        verts = [np.asarray(trimesh.load_mesh(f, process=False).vertices, float) for f in files]
        V = np.vstack(verts) if verts else np.zeros((0, 3))
        if len(V) > max_verts:                       # 顶点抽样，净空 12mm 下足够
            V = V[::len(V) // max_verts]
        self.tree = cKDTree(V) if len(V) else None

    def dist(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        return self.tree.query(pts, k=1)[0] if self.tree else np.full(len(pts), np.inf)

    def free_seg(self, a, b, c):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 6)))
        return bool((self.dist(a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]) >= c).all())


class SurfacePreference:
    """Finite selected sheet domains, offset toward the declared routing side.

    Deterministic triangle subdivision bounds sampling gaps by spacing mm.
    This distance is a routing preference, not an obstacle clearance certificate.
    """
    def __init__(self, parts, clearance, spacing):
        samples = []
        for part in parts:
            mesh = trimesh.load_mesh(part['path'], process=False)
            vertices, _ = trimesh.remesh.subdivide_to_size(
                mesh.vertices, mesh.faces, max_edge=spacing, max_iter=20)
            samples.append(np.unique(vertices, axis=0) + clearance * unit(part['direction']))
        self.tree = cKDTree(np.vstack(samples))

    def dist(self, points):
        return self.tree.query(np.asarray(points, float), k=1)[0]
