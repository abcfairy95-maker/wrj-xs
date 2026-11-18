"""分叉点几何约束（两个方法共享）：l1/l2/l5/l6/x 的读取与检测。

- l1 末端分支长度：终点到最近分叉 ≥ l1
- l2 分叉点间距：相邻分叉弧长 ≥ l2
- l5 卡扣到分叉距离：卡扣到最近分叉 ≥ l5
- l6 卡扣到起终点距离：卡扣到起点/终点 ≥ l6
- x  最大单点分支数量：分叉点 degree ≤ x
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def read_fork_params(case):
    prefs = case['prefs']
    return dict(
        l1=float(prefs.get('terminal_branch_min_length_mm', 60)),
        l2=float(prefs.get('split_point_min_spacing_mm', 75)),
        l5=float(prefs.get('clip_min_distance_to_split_mm', 40)),
        l6=float(prefs.get('clip_terminal_min_distance_mm', 60)),
        x=int(prefs.get('max_branch_count', 4)),
    )


def _segment_graph(topology):
    R, C, W = [], [], []
    for seg in topology['segments']:
        pts = np.asarray(topology['edge_paths'][str(seg['id'])]['path_mm'], float)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        a, b = seg['from_cp_id'], seg['to_cp_id']
        R += [a, b]; C += [b, a]; W += [L, L]
    n = max(max(R, default=0), max(C, default=0)) + 1
    return csr_matrix((W, (R, C)), shape=(n, n))


def detect_fork_violations(topology, params):
    """检测 l1/l2/l5/l6/x 违规，返回违规清单（只读，不修改）。"""
    cps = topology['control_points']
    branches = [cp for cp in cps if cp['type'] == 'branch']
    starts = [cp['id'] for cp in cps if cp['type'] == 'start']
    ends = [h['end_cp_id'] for h in topology['harnesses']]
    clips = [c['control_point_id'] for c in topology['clip_points']]
    g = _segment_graph(topology)
    violations = []

    # x：分叉度数超限
    for b in branches:
        if b.get('degree', 0) > params['x']:
            violations.append(dict(constraint='x', fork_cp_id=int(b['id']),
                                   degree=int(b.get('degree', 0)), limit=int(params['x'])))

    branch_ids = [b['id'] for b in branches]

    # l1：终点到最近分叉
    if ends and branch_ids:
        d = dijkstra(g, directed=False, indices=[int(e) for e in ends])[:, [int(i) for i in branch_ids]]
        for i, e in enumerate(ends):
            mn = float(d[i].min()) if np.isfinite(d[i]).any() else np.inf
            if mn < params['l1']:
                violations.append(dict(constraint='l1', end_cp_id=int(e),
                                       arc_to_fork_mm=round(mn, 2), limit=float(params['l1'])))

    # l5：卡扣到最近分叉
    if clips and branch_ids:
        d = dijkstra(g, directed=False, indices=[int(c) for c in clips])[:, [int(i) for i in branch_ids]]
        for i, c in enumerate(clips):
            mn = float(d[i].min()) if np.isfinite(d[i]).any() else np.inf
            if mn < params['l5']:
                violations.append(dict(constraint='l5', clip_cp_id=int(c),
                                       arc_to_fork_mm=round(mn, 2), limit=float(params['l5'])))

    # l2：相邻分叉间距（两分叉间无其他分叉的近似：直接测两两弧长）
    if len(branch_ids) >= 2:
        d = dijkstra(g, directed=False, indices=[int(i) for i in branch_ids])[:, [int(i) for i in branch_ids]]
        for i in range(len(branch_ids)):
            for j in range(i + 1, len(branch_ids)):
                if np.isfinite(d[i, j]) and d[i, j] < params['l2']:
                    violations.append(dict(constraint='l2', fork_a=int(branch_ids[i]),
                                           fork_b=int(branch_ids[j]),
                                           arc_mm=round(float(d[i, j]), 2), limit=float(params['l2'])))

    # l6：卡扣到起点/终点
    terminals = starts + ends
    if clips and terminals:
        d = dijkstra(g, directed=False, indices=[int(c) for c in clips])[:, [int(t) for t in terminals]]
        for i, c in enumerate(clips):
            mn = float(d[i].min()) if np.isfinite(d[i]).any() else np.inf
            if mn < params['l6']:
                violations.append(dict(constraint='l6', clip_cp_id=int(c),
                                       arc_to_terminal_mm=round(mn, 2), limit=float(params['l6'])))

    return violations
