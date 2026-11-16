"""平滑层：CPU（碰撞捷径 + 三次 Hermite）与 GPU（PyTorch Adam 可微分）两套算子。"""
import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def shortcut(pts, fixed, col, c):
    pts, fixed = [np.asarray(p, float) for p in pts], list(fixed)
    i, n = 0, len(pts)
    while i < n - 2:
        j = i + 1
        while j < n and not fixed[j]:
            j += 1
        k = min(j, n - 1)
        while k > i + 1:
            if col.free_seg(pts[i], pts[k], c):
                del pts[i + 1:k]; del fixed[i + 1:k]
                n = len(pts); break
            k -= 1
        i += 1
    return pts, fixed


def enforce_tangents(pts, fixed, tg, anchor=30.0):
    for idx, t, mode in sorted(tg, key=lambda x: -x[0]):
        p = pts[idx]; t = unit(t); L = anchor
        for s in (-1, 1):
            j = idx + s
            while 0 <= j < len(pts) and not fixed[j]:
                j += s
            if 0 <= j < len(pts) and j != idx:
                L = min(L, 0.45 * np.linalg.norm(pts[j] - p))
        if mode in ('both', 'after'):
            pts[idx + 1:idx + 1] = [p + t * L]; fixed[idx + 1:idx + 1] = [True]
        if mode in ('both', 'before'):
            pts[idx:idx] = [p - t * L]; fixed[idx:idx] = [True]
    return pts, fixed


def dedupe(pts, tol=1.0, fixed=None):
    out = [np.asarray(pts[0], float)]
    for i, p in enumerate(pts[1:], 1):
        distance = np.linalg.norm(p - out[-1])
        if distance > tol or (fixed is not None and fixed[i] and distance > 0):
            out.append(np.asarray(p, float))
    return out


def resample(pts, step=3.0):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], max(2, int(cum[-1] / step) + 1))
    return np.stack([np.interp(s, cum, pts[:, d]) for d in range(3)], -1)


def smooth_path(pts, step=3.0):
    """CatmullRom 平滑重采样（与 viewer 一致），把折线变光滑曲线。"""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return resample(pts, step)
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p0 = pts[max(0, i - 1)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(len(pts) - 1, i + 2)]
        L = np.linalg.norm(p2 - p1)
        n = max(2, int(L / step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            t2 = t * t; t3 = t2 * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def hermite_smooth(pts, tang, step=3.0):
    """三次 Hermite 插值：过关键点且强制切线，得到 C1 光滑曲线（与参考一致）。"""
    from scipy.interpolate import CubicHermiteSpline
    pts = np.asarray(pts, float); tang = np.asarray(tang, float)
    if len(pts) < 2:
        return resample(pts, step)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return pts
    s = np.linspace(0, cum[-1], max(2, int(cum[-1] / step) + 1))
    s = np.unique(np.concatenate([s, cum]))
    out = np.empty((len(s), 3))
    for d in range(3):
        out[:, d] = CubicHermiteSpline(cum, pts[:, d], tang[:, d])(s)
    return out


def smooth_segment(coords, t0, t1, col, clearance, step=3.0):
    """对单条段的原始网格路径做捷径 + Hermite 平滑，端点切线 t0/t1（None 则用边方向）。"""
    coords = np.asarray(coords, float)
    if len(coords) < 2:
        return coords
    fixed = [True] + [False] * (len(coords) - 2) + [True]
    pts, _ = shortcut(list(coords), fixed, col, clearance)
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return coords
    tang = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        if i == 0:
            tang[i] = t0 if (t0 is not None and np.any(t0)) else unit(pts[1] - pts[0])
        elif i == len(pts) - 1:
            tang[i] = t1 if (t1 is not None and np.any(t1)) else unit(pts[-1] - pts[-2])
        else:
            d1 = unit(pts[i] - pts[i - 1]); d2 = unit(pts[i + 1] - pts[i])
            t = d1 + d2; tang[i] = t / (np.linalg.norm(t) or 1)
    return hermite_smooth(pts, tang, step)


def smooth_gpu(points, clip_indices, clip_dirs, device, steps=200, fixed_indices=()):
    """GPU 可微分平滑：Adam 优化控制点，Elastic 拉直 + 卡扣切向硬对齐。

    points: [N,3] 折线控制点（卡扣点已精确）；clip_indices/dirs：卡扣下标与切向。
    """
    import torch
    P = torch.tensor(np.asarray(points, float), dtype=torch.float32, device=device)
    fixed = torch.tensor(np.asarray(points, float), dtype=torch.float32, device=device)
    clip_i = list(clip_indices)
    locked = sorted(set(fixed_indices) | set(clip_i) | {0, len(points) - 1})
    clip_d = torch.tensor(np.asarray(clip_dirs, float), dtype=torch.float32, device=device)
    P.requires_grad_(True)
    opt = torch.optim.Adam([P], lr=0.5)

    def elastic(x):
        d2 = x[:-2] - 2 * x[1:-1] + x[2:]
        return (d2 * d2).sum()

    def tangent(x):
        if not clip_i:
            return torch.tensor(0.0, device=device)
        loss = 0.0
        for k, d in zip(clip_i, clip_d):
            if k <= 0 or k >= len(x) - 1:
                continue
            flow = x[k + 1] - x[k - 1]
            n = flow.norm() + 1e-9
            loss += 1.0 - torch.dot(flow / n, d).abs()
        return loss

    for _ in range(steps):
        opt.zero_grad()
        loss = elastic(P) + 1000.0 * tangent(P)
        loss.backward()
        P.grad[locked] = 0
        opt.step()
        with torch.no_grad():
            P[locked] = fixed[locked]
    result = P.detach().cpu().numpy().astype(float)
    result[locked] = np.asarray(points, float)[locked]
    return result


def smooth_network(rm, paths, protected_nodes, directions, device='cpu', step=3.0):
    """Jointly smooth complete routes with shared coordinates and C1 derivatives.

    directions maps protected node IDs to prescribed axes. Returns whole-route
    samples and one sample offset for every node occurrence in every route.
    """
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import CubicHermiteSpline

    node_ids = sorted({node for path in paths.values() for node in path})
    if not node_ids:
        return {}, {}
    index = {node: i for i, node in enumerate(node_ids)}
    original = np.asarray([rm.pos(node) for node in node_ids], float)
    locked = np.array([node in protected_nodes for node in node_ids])
    triples = sorted({(min(a, c), b, max(a, c)) for path in paths.values()
                      for a, b, c in zip(path, path[1:], path[2:])})
    rows, columns, values = [], [], []
    for row, (a, b, c) in enumerate(triples):
        rows.extend([row] * 3)
        columns.extend([index[a], index[b], index[c]])
        values.extend([1., -2., 1.])
    difference = coo_matrix((values, (rows, columns)), shape=(len(triples), len(node_ids))).tocsr()
    # Preserve the preferred-surface route during whole-network smoothing.
    fidelity = 1.0 if getattr(rm, 'surface_preference', None) is not None else 0.01
    positions = original.copy()
    free = ~locked
    if free.any() and device == 'cpu':
        system = difference.T @ difference + fidelity * eye(len(node_ids), format='csr')
        rhs = fidelity * original[free] - system[free][:, locked] @ original[locked]
        positions[free] = spsolve(system[free][:, free].tocsc(), rhs)
    elif free.any():
        import torch
        backend = 'cuda' if torch.cuda.is_available() else 'cpu'
        points = torch.tensor(original, dtype=torch.float64, device=backend, requires_grad=True)
        reference = points.detach().clone()
        fixed = torch.tensor(locked, device=backend)
        triplet_indices = torch.tensor([[index[n] for n in triple] for triple in triples],
                                       dtype=torch.long, device=backend).reshape(-1, 3)
        optimizer = torch.optim.Adam([points], lr=0.5)
        for _ in range(200):
            optimizer.zero_grad()
            curvature = (points[triplet_indices[:, 0]] - 2 * points[triplet_indices[:, 1]]
                         + points[triplet_indices[:, 2]])
            loss = curvature.square().sum() + fidelity * (points - reference).square().sum()
            loss.backward()
            points.grad[fixed] = 0
            optimizer.step()
            with torch.no_grad():
                points[fixed] = reference[fixed]
        positions = points.detach().cpu().numpy()
    positions[locked] = original[locked]

    # At each visit the incoming and outgoing outward derivatives are opposite.
    # Solve these signed relations jointly so shared geometry cannot diverge.
    neighbours = {node: set() for node in node_ids}
    relations = {node: {} for node in node_ids}
    for path in paths.values():
        for a, b in zip(path, path[1:]):
            neighbours[a].add(b)
            neighbours[b].add(a)
        for a, b, c in zip(path, path[1:], path[2:]):
            relations[b].setdefault(a, set()).add(c)
            relations[b].setdefault(c, set()).add(a)
    directed_neighbours = {}
    for path in paths.values():
        directed_neighbours.setdefault(path[0], set()).add(path[1])
        directed_neighbours.setdefault(path[-1], set()).add(path[-2])
    outward = {}
    for node in node_ids:
        remaining = set(neighbours[node])
        while remaining:
            first = min(remaining)
            signs = {first: 1}
            stack = [first]
            conflict = False
            while stack:
                neighbour = stack.pop()
                for other in relations[node].get(neighbour, ()):
                    expected = -signs[neighbour]
                    if other in signs:
                        conflict |= signs[other] != expected
                    else:
                        signs[other] = expected
                        stack.append(other)
            remaining.difference_update(signs)
            axis = np.sum([sign * unit(positions[index[other]] - positions[index[node]])
                           for other, sign in sorted(signs.items())], axis=0)
            if node in directions:
                prescribed = unit(directions[node])
                terminal_neighbours = sorted(directed_neighbours.get(node, set()) & signs.keys())
                if terminal_neighbours:
                    axis = prescribed * signs[terminal_neighbours[0]]
                else:
                    axis = prescribed if np.dot(axis, prescribed) >= 0 else -prescribed
            derivative = np.zeros(3) if conflict else unit(axis)
            for other, sign in signs.items():
                outward[node, other] = sign * derivative

    geometries, sample_offsets = {}, {}
    for endpoint, path in paths.items():
        points = positions[[index[node] for node in path]]
        tangents = np.asarray([outward[a, b] for a, b in zip(path, path[1:])]
                              + [-outward[path[-1], path[-2]]])
        lengths = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1), 1e-9)
        knots = np.concatenate([[0.], np.cumsum(lengths)])
        # One spline spans the whole route. Each interval uses the same sampling
        # density in all routes, allowing exact occurrence-based export slicing.
        counts = np.maximum(1, np.ceil(lengths / step).astype(int))
        parameters = np.concatenate([np.linspace(knots[i], knots[i + 1], count + 1)[:-1]
                                     for i, count in enumerate(counts)] + [knots[-1:]])
        geometry = CubicHermiteSpline(knots, points, tangents)(parameters)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        geometry[offsets] = points
        geometries[endpoint] = geometry
        sample_offsets[endpoint] = offsets
    return geometries, sample_offsets
