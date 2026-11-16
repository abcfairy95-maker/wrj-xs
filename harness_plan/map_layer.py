"""地图层：体素自由空间路图（网格 + 终端接入 + 贴面偏好），含 device 分发。

Roadmap 是唯一的数据结构，同时服务 CPU 与 GPU 两套寻路：
- CPU：6 邻接加权图 + scipy Dijkstra（精确终端坐标 + 卡扣方向代价 + 贴面偏好）。
- GPU：体素网格上并行波前传播（终端吸附到最近自由体素，6 连通 BFS）。
"""
import numpy as np
from scipy.sparse import csr_matrix

from . import pathfind_layer as pf
from .data_layer import unit


class Roadmap:
    def __init__(self, col, lo, hi, vs, clearance, device='cpu'):
        lo = np.floor(lo / vs) * vs; hi = np.ceil(hi / vs) * vs
        self.vs, self.clearance = vs, clearance
        self.origin = lo
        self.shape = tuple(int((hi[d] - lo[d]) / vs) + 1 for d in range(3))
        g = [lo[d] + (np.arange(self.shape[d]) + .5) * vs for d in range(3)]
        gp = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
        free = col.dist(gp) >= clearance
        self.idx = np.full(self.shape, -1, np.int64)
        self.idx[free.reshape(self.shape)] = np.arange(int(free.sum()))
        self.nodes = [gp[free]]
        self.edges = []
        self.clip_dir = {}
        self._g = None
        self.device = device
        # 节点序号 → 体素坐标（仅自由体素节点；终端节点在 add_terminals 时吸附补录）
        self.node_vox = {}
        for v in np.argwhere(self.idx >= 0):
            self.node_vox[int(self.idx[tuple(v)])] = tuple(int(x) for x in v)

    def set_preference(self, surf, weight):
        """Use the actual selected sheet domains, with no ordering between parts."""
        self.pref = weight * surf.dist(self.nodes[0])
        self.surface_preference = surf

    @property
    def n(self):
        return sum(len(x) for x in self.nodes)

    def all(self):
        return np.vstack(self.nodes)

    def pos(self, i):
        for x in self.nodes:
            if i < len(x):
                return x[i]
            i -= len(x)

    def _nearest_vox(self, pos):
        """把任意坐标吸附到最近自由体素（供 GPU 波前传播用）。"""
        v = np.floor((np.asarray(pos, float) - self.origin) / self.vs).astype(int)
        v = np.clip(v, 0, np.array(self.shape) - 1)
        if self.idx[tuple(v)] >= 0:
            return tuple(int(x) for x in v)
        best, best_d = tuple(v), np.inf
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nv = (int(v[0] + dz), int(v[1] + dy), int(v[2] + dx))
                    if all(0 <= nv[d] < self.shape[d] for d in range(3)) and self.idx[nv] >= 0:
                        d = float(np.linalg.norm(np.array(nv) - v))
                        if d < best_d:
                            best_d, best = d, nv
        return best

    def grid_edges(self):
        idx, vs, R, C, W = self.idx, self.vs, [], [], []
        for ax in range(3):
            for s in (1,):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                (a if s == 1 else b)[ax] = slice(0, -1)
                (b if s == 1 else a)[ax] = slice(1, None)
                A, B = idx[tuple(a)], idx[tuple(b)]
                ok = (A >= 0) & (B >= 0)
                i, j = A[ok].astype(np.int64), B[ok].astype(np.int64)
                w = np.full(i.shape, float(vs))
                if getattr(self, 'pref', None) is not None:   # 贴面偏好
                    w = w + (self.pref[i] + self.pref[j]) / 2.0
                R += [i, j]; C += [j, i]; W += [w, w]
        self.edges.append((np.concatenate(R), np.concatenate(C), np.concatenate(W)))

    def add_terminals(self, pts, dirs=None):
        start = self.n
        pts = np.atleast_2d(np.asarray(pts, float))
        self.nodes.append(pts)
        g, R, C, W = self.nodes[0], [], [], []
        for t, p in enumerate(pts):
            ti = start + t
            ds = np.linalg.norm(g - p, axis=1)
            for gi in np.argsort(ds)[:12]:
                if ds[gi] > 4 * self.vs:
                    break
                R += [ti, gi]; C += [gi, ti]; W += [float(ds[gi])] * 2
            if dirs is not None and dirs[t] is not None:
                self.clip_dir[ti] = unit(dirs[t])
            self.node_vox[ti] = self._nearest_vox(p)
        self.edges.append((np.asarray(R), np.asarray(C), np.asarray(W, float)))
        self._g = None
        return list(range(start, start + len(pts)))

    def graph(self, turn_w=30.0):
        if self._g is None:
            R = np.concatenate([e[0] for e in self.edges])
            C = np.concatenate([e[1] for e in self.edges])
            W = np.concatenate([e[2] for e in self.edges]).astype(float)
            self._g = csr_matrix((W, (R, C)), shape=(self.n, self.n))
            if self.clip_dir:
                P = self.all()
                data = self._g.data.copy()
                for node, d in self.clip_dir.items():
                    s, e = self._g.indptr[node], self._g.indptr[node + 1]
                    nbrs = self._g.indices[s:e]
                    delta = P[nbrs] - P[node]
                    nrm = np.linalg.norm(delta, axis=1)
                    cos = np.zeros(len(nbrs))
                    ok = nrm > 1e-9
                    cos[ok] = np.abs(np.sum(delta[ok] / nrm[ok, None] * d, axis=1))
                    data[s:e] += turn_w * (1.0 - cos)
                self._g = csr_matrix((data, self._g.indices, self._g.indptr),
                                     shape=self._g.shape)
        return self._g

    def _free_grid(self):
        return self.idx >= 0

    def path(self, s, t):
        if s == t:
            return [int(s)]
        if self.device == 'gpu':
            return self._path_gpu(s, t)
        if getattr(self, 'use_astar', False):
            return pf.astar(self.graph(), self.all(), s, t)
        return pf.dijkstra_path(self.graph(), s, t)

    def pair(self, terms):
        if self.device == 'gpu':
            return self._pair_gpu(terms)
        return pf.dijkstra_pair(self.graph(), terms)

    # ---- GPU 通道：波前传播（节点序号 ↔ 体素坐标映射） ----
    def _voxel_cost(self):
        cost = np.full(self.shape, float(self.vs))
        if getattr(self, 'pref', None) is not None:
            cost[self.idx >= 0] += self.pref
        return cost

    def _pair_gpu(self, terms):
        import torch
        free = self._free_grid()
        src_vox = [self.node_vox.get(int(t), self._nearest_vox(self.pos(int(t)))) for t in terms]
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        dist = pf.wavefront_batch(free, src_vox, dev, cost=self._voxel_cost()).cpu().numpy()
        self._distance_fields = {voxel: field for voxel, field in zip(src_vox, dist)}
        B = len(terms)
        D = np.full((B, B), np.inf)
        for a in range(B):
            for b in range(B):
                D[a, b] = dist[a][src_vox[b]]
        return D

    def _path_gpu(self, s, t):
        import torch
        free = self._free_grid()
        sv = self.node_vox.get(int(s), self._nearest_vox(self.pos(int(s))))
        tv = self.node_vox.get(int(t), self._nearest_vox(self.pos(int(t))))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        dist = getattr(self, '_distance_fields', {}).get(sv)
        if dist is None:
            dist = pf.wavefront_batch(free, [sv], dev, cost=self._voxel_cost()).cpu().numpy()[0]
        voxpath = pf.backtrack(dist, sv, tv, free, cost=self._voxel_cost())
        if voxpath is None:
            return None
        nodes = []
        for v in voxpath:
            node = int(self.idx[v])
            if node >= 0 and (not nodes or nodes[-1] != node):
                nodes.append(node)
        if not nodes or nodes[0] != int(s):
            nodes = [int(s)] + nodes
        if nodes[-1] != int(t):
            nodes = nodes + [int(t)]
        return nodes
