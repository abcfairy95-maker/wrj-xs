"""寻路层：CPU（Dijkstra/A*）与 GPU（并行波前传播）两套算子，统一返回节点序号路径。

约定：所有函数工作在「节点序号」空间（node index），由地图层 Roadmap 提供节点坐标。
GPU 波前传播在体素网格上运行，地图层负责体素↔节点序号映射。
"""
import heapq

import numpy as np
from scipy.sparse.csgraph import dijkstra


# ================= CPU：A* =================
def astar(g, P, s, t):
    """A* 单源单目标（欧氏启发）。返回节点序列或 None。"""
    h = lambda u: float(np.linalg.norm(P[u] - P[t]))
    dist = {int(s): 0.0}
    prev = {}
    pq = [(h(int(s)), 0.0, int(s))]
    while pq:
        _, d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u == int(t):
            break
        for v, w in zip(g.indices[g.indptr[u]:g.indptr[u + 1]], g.data[g.indptr[u]:g.indptr[u + 1]]):
            v = int(v); nd = d + float(w)
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd + h(v), nd, v))
    if int(t) not in dist or not np.isfinite(dist[int(t)]):
        return None
    out, cur = [int(t)], int(t)
    while cur != int(s):
        cur = prev[cur]; out.append(cur)
    return out[::-1]


# ================= CPU：Dijkstra =================
def dijkstra_path(g, s, t):
    if s == t:
        return [int(s)]
    dist, pred = dijkstra(g, indices=[int(s)], return_predecessors=True)
    if not np.isfinite(dist[0][int(t)]):
        return None
    out, cur = [int(t)], int(t)
    while cur != int(s):
        cur = pred[0][cur]
        if cur < 0:
            return None
        out.append(cur)
    return out[::-1]


def dijkstra_pair(g, terms):
    D = np.full((len(terms), len(terms)), np.inf)
    for a, s in enumerate(terms):
        dd = dijkstra(g, indices=[int(s)])[0]
        for b, t in enumerate(terms):
            D[a, b] = dd[int(t)]
    return D


# ================= GPU：并行波前传播 =================
def wavefront_batch(free, sources, device, max_iter=None, cost=None):
    """从每个源并行波前扩散，返回 [B, D, H, W] 距离场。

    free: (D,H,W) 布尔自由场；sources: [(dz,dy,dx), ...] 体素坐标。
    """
    import torch
    free = torch.as_tensor(free, dtype=torch.bool, device=device)
    D, H, W = free.shape
    B = len(sources)
    INF = float('inf')
    dist = torch.full((B, D, H, W), INF, device=device, dtype=torch.float32)
    for b, s in enumerate(sources):
        if free[s[0], s[1], s[2]]:
            dist[b, s[0], s[1], s[2]] = 0.0
    f = free.unsqueeze(0)
    cost = torch.ones_like(free, dtype=torch.float32) if cost is None else torch.as_tensor(
        cost, dtype=torch.float32, device=device)
    if not bool(torch.all(torch.isfinite(cost) & (cost > 0))):
        raise ValueError('Wavefront costs must be positive and finite')
    padded_cost = torch.nn.functional.pad(cost, (1, 1, 1, 1, 1, 1), value=0)
    neighbour_cost = torch.stack([
        padded_cost[:-2, 1:-1, 1:-1], padded_cost[2:, 1:-1, 1:-1],
        padded_cost[1:-1, :-2, 1:-1], padded_cost[1:-1, 2:, 1:-1],
        padded_cost[1:-1, 1:-1, :-2], padded_cost[1:-1, 1:-1, 2:]])
    edge_cost = (neighbour_cost + cost) / 2

    max_iter = D * H * W if max_iter is None else max_iter
    for _ in range(max_iter):
        p = torch.nn.functional.pad(dist, (1, 1, 1, 1, 1, 1), value=INF)
        n = torch.stack([
            p[:, :-2, 1:-1, 1:-1], p[:, 2:, 1:-1, 1:-1],
            p[:, 1:-1, :-2, 1:-1], p[:, 1:-1, 2:, 1:-1],
            p[:, 1:-1, 1:-1, :-2], p[:, 1:-1, 1:-1, 2:],
        ])
        nd = torch.minimum(dist, (n + edge_cost[:, None]).min(dim=0)[0])
        nd = nd.masked_fill(~f, INF)
        if torch.equal(nd, dist):
            break
        dist = nd
    return dist


def backtrack(dist_field, source, target, free_np, cost=None):
    """从 target 沿距离下降回溯到 source，返回体素坐标序列。"""
    D, H, W = dist_field.shape
    if not free_np[source] or not free_np[target] or not np.isfinite(dist_field[target]):
        return None
    path = [target]
    cur = target
    guard = 0
    while cur != source and guard < D * H * W:
        guard += 1
        best, best_d = None, np.inf
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)):
            nb = (cur[0] + dz, cur[1] + dy, cur[2] + dx)
            if 0 <= nb[0] < D and 0 <= nb[1] < H and 0 <= nb[2] < W:
                edge = 1.0 if cost is None else (cost[cur] + cost[nb]) / 2
                candidate = float(dist_field[nb]) + edge
                if free_np[nb] and dist_field[nb] < dist_field[cur] and candidate < best_d:
                    best, best_d = nb, candidate
        if best is None:
            return None
        cur = best
        path.append(cur)
    return path[::-1] if cur == source else None
