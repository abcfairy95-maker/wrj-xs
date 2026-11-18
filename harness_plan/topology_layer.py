"""拓扑层：MST 主干 + 终点挂叶子 + 卡扣强穿（CPU/GPU 共享，图逻辑不适合 GPU）。"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree


def _tree_graph(rm, paths):
    """把线束路径并集成带权树图，返回 (adj, node_list, idx, G)。边权=相邻体素欧氏距离。"""
    edges = set()
    for p in paths.values():
        for u, v in zip(p[:-1], p[1:]):
            edges.add((min(u, v), max(u, v)))
    node_list = sorted({n for e in edges for n in e})
    idx = {n: i for i, n in enumerate(node_list)}
    adj = {n: [] for n in node_list}
    R, C, W = [], [], []
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        w = float(np.linalg.norm(rm.pos(u) - rm.pos(v)))
        R += [idx[u], idx[v]]
        C += [idx[v], idx[u]]
        W += [w, w]
    n = len(node_list)
    G = csr_matrix((W, (R, C)), shape=(n, n)) if n else csr_matrix((0, 0))
    return adj, node_list, idx, G


def _multisource(G, idx, sources):
    """多源最短路：返回每个节点到最近 source 的弧长（无 source 则 inf）。"""
    src = [idx[s] for s in sources if s in idx]
    if not src:
        return {n: np.inf for n in idx}
    d = dijkstra(G, directed=False, indices=src, min_only=True)
    return {n: d[idx[n]] for n in idx}


def _edges_to_graph(rm, edges):
    """边集合 -> (node_list, idx, 带权图)。边权 = 相邻体素欧氏距离。"""
    node_list = sorted({n for e in edges for n in e})
    idx = {n: i for i, n in enumerate(node_list)}
    R, C, W = [], [], []
    for u, v in edges:
        w = float(np.linalg.norm(rm.pos(u) - rm.pos(v)))
        R += [idx[u], idx[v]]
        C += [idx[v], idx[u]]
        W += [w, w]
    n = len(node_list)
    G = csr_matrix((W, (R, C)), shape=(n, n)) if n else csr_matrix((0, 0))
    return node_list, idx, G


def _root_parent(adj, root):
    parent = {root: -1}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                stack.append(v)
    return parent


def _position_forks(rm, paths, nodes, all_terminals, n_end, params):
    """分叉点定位：把违规分叉点滑移（合并到更合适的邻居），满足 l1/l2/l5/x。

    分叉点是自由的（可沿主干移动），卡扣/终点是固定的。所以让分叉点
    「迎合」卡扣/终点/其他分叉移动，直到满足：
      l5 卡扣到分叉 ≥ l5、l1 终点到分叉 ≥ l1、l2 分叉间距 ≥ l2、x 度数 ≤ x。
    """
    l1, l2, l5, x = params['l1'], params['l2'], params['l5'], params['x']
    clip_nodes = list(nodes[1 + n_end:])
    end_nodes = list(nodes[1:1 + n_end])
    _env = __import__('os').environ
    if _env.get('NO_FORK_POSITION'):
        return paths
    _skip = {'l1': _env.get('SKIP_L1'), 'l2': _env.get('SKIP_L2'), 'l5': _env.get('SKIP_L5'), 'x': _env.get('SKIP_X')}
    for _ in range(80):
        adj, node_list, idx, G = _tree_graph(rm, paths)
        if not adj:
            break
        forks = [n for n in node_list if len(adj[n]) >= 3 and n not in all_terminals]
        if not forks:
            break
        d_clip = _multisource(G, idx, clip_nodes)      # 节点到最近卡扣弧长
        d_end = _multisource(G, idx, end_nodes)        # 节点到最近终点弧长
        dist = {F: dijkstra(G, directed=False, indices=[idx[F]], min_only=True) for F in forks}
        moved = False
        for F in forks:
            deg = len(adj[F])
            dF_clip, dF_end = d_clip[F], d_end[F]
            others = [o for o in forks if o != F]
            dF_fork = min((dist[F][idx[o]] for o in others), default=np.inf)
            N = None
            if not _skip['x'] and deg > x:
                # x：度数超限，合并到最近的叶子邻居（降一度）
                leaves = [n for n in adj[F] if len(adj[n]) == 1]
                cand = leaves or adj[F]
                N = min(cand, key=lambda n: np.linalg.norm(rm.pos(n) - rm.pos(F)))
            elif not _skip['l5'] and dF_clip < l5:
                # l5：离卡扣太近，合并到离卡扣最远的邻居（避免合并后度数超 x）
                cand = [n for n in adj[F] if len(adj[n]) + deg - 2 <= x]
                if cand:
                    N = max(cand, key=lambda n: d_clip[n])
                    if d_clip[N] <= dF_clip:
                        N = None
                else:
                    N = None
            elif not _skip['l1'] and dF_end < l1:
                # l1：离终点太近（末端分支太短），合并到离终点最远的邻居（避免超 x）
                cand = [n for n in adj[F] if len(adj[n]) + deg - 2 <= x]
                if cand:
                    N = max(cand, key=lambda n: d_end[n])
                    if d_end[N] <= dF_end:
                        N = None
                else:
                    N = None
            elif not _skip['l2'] and dF_fork < l2:
                # l2：分叉间距太近。可合并则「合并近分叉」（零膨胀）；
                # 合并会超 x 时「拉开」：把分叉朝 other 反方向合并到穿行点邻居
                # （穿行点变分叉，degree 不变，不超 x，且把间距拉开）。
                other = min(others, key=lambda o: dist[F][idx[o]])
                if deg + len(adj[other]) - 2 <= x:
                    N = min(adj[F], key=lambda n: dist[other][idx[n]])
                    if dist[other][idx[N]] >= dist[other][idx[F]]:
                        N = None
                else:
                    cand = [n for n in adj[F] if len(adj[n]) == 2 and n not in all_terminals]
                    if cand:
                        N = max(cand, key=lambda n: dist[other][idx[n]])
                        if dist[other][idx[N]] <= dist[other][idx[F]]:
                            N = None
                    else:
                        N = None
            else:
                continue
            if N is None:
                continue
            for eid, p in list(paths.items()):
                if F not in p:
                    continue
                i = p.index(F)
                new_p = list(p)
                if (i > 0 and new_p[i - 1] == N) or (i < len(new_p) - 1 and new_p[i + 1] == N):
                    del new_p[i]
                else:
                    new_p[i] = N
                paths[eid] = new_p
            moved = True
        if not moved:
            break
    return paths


def steiner(rm, case):
    """方法三（终点先行）：MST{start+ends} 终点骨架定接入顺序，卡扣主动插入穿行。

    约束：终点是叶子（degree=1）、卡扣是穿行点（degree=2）、分叉只落在自由体素。
    """
    ends, clips = case['ends'], case['clips']
    n_end = len(ends)

    # 1. 终端全部进图；顺序固定为 start + ends + clips（与结果组装索引一致）
    #    终点/卡扣用低连通度接入，从图结构上杜绝「终端被路径穿过变分叉」：
    #    start=12 连通（根/主干），end=1 连通（叶子），clip=2 连通（穿行）
    start_term = rm.add_terminals([case['start']['pos']], n_connect=12)[0]
    end_terms = rm.add_terminals([e['pos'] for e in ends], n_connect=1)
    clip_terms = rm.add_terminals([c['pos'] for c in clips], [c['dir'] for c in clips],
                                  n_connect=2)
    nodes = [start_term] + end_terms + clip_terms
    all_terminals = set(nodes)

    # 2. MST {start + ends} 草稿：终点先行，骨架由终点决定接入顺序
    from .fork_constraints import read_fork_params
    _fp = read_fork_params(case)
    _x, _l1, _l2 = _fp['x'], _fp['l1'], _fp['l2']
    trunk = [start_term] + end_terms
    D = rm.pair(trunk)
    mst = minimum_spanning_tree(csr_matrix(np.where(np.isfinite(D), D, 0.0))).tocoo()
    draft = {}
    for a, b in zip(mst.row, mst.col):
        draft.setdefault(int(a), []).append(int(b))
        draft.setdefault(int(b), []).append(int(a))
    order, seen, q = [], {0}, [0]
    while q:
        u = q.pop(0)
        for v in sorted(draft.get(u, ())):
            if v not in seen:
                seen.add(v); order.append(v - 1); q.append(v)
    order += [i for i in range(n_end) if i not in order]
    edges = set()
    # 3. 终点按 MST 顺序挂叶子（终点=叶子 degree=1，不参与主干分叉）
    deg = {}
    for idx in order:
        en = end_terms[idx]
        if not edges:
            p = rm.path(start_term, en)
        else:
            tree_nodes = {start_term}
            for u, v in edges:
                tree_nodes.add(u); tree_nodes.add(v)
            dist, pred = dijkstra(rm.graph(), indices=[en], return_predecessors=True)
            dist = dist[0]; pred = pred[0]
            _nl, _ix, _G = _edges_to_graph(rm, edges)
            forks_now = [n for n in _nl if deg.get(n, 0) >= 3 and n not in all_terminals]
            fork_set = set(forks_now)
            d_fork_tree = _multisource(_G, _ix, forks_now)

            def first_fork_euclid(tn):
                d_tn = float(np.linalg.norm(rm.pos(en) - rm.pos(tn)))
                cur = tn
                while cur != en and cur >= 0:
                    if cur in fork_set and cur != tn:
                        return min(d_tn, float(np.linalg.norm(rm.pos(en) - rm.pos(cur))))
                    cur = int(pred[cur])
                return d_tn

            best, best_d = None, np.inf
            _l1_need = _l1 + 2.0
            for f in forks_now:
                if deg.get(f, 0) >= _x or first_fork_euclid(f) < _l1_need:
                    continue
                if dist[f] < best_d:
                    best_d, best = dist[f], f
            for tn in tree_nodes:
                if tn in all_terminals or deg.get(tn, 0) >= _x:
                    continue
                if first_fork_euclid(tn) < _l1_need:
                    continue
                if deg.get(tn, 0) < 3 and d_fork_tree.get(tn, np.inf) < _l2:
                    continue
                if dist[tn] < best_d:
                    best_d, best = dist[tn], tn
            if best is None:
                for f in forks_now:
                    if deg.get(f, 0) >= _x:
                        continue
                    if dist[f] < best_d:
                        best_d, best = dist[f], f
                for tn in tree_nodes:
                    if tn in all_terminals or deg.get(tn, 0) >= _x:
                        continue
                    if dist[tn] < best_d:
                        best_d, best = dist[tn], tn
            p = None
            if best is not None and np.isfinite(best_d):
                path = [best]; cur = best
                while cur != en:
                    cur = int(pred[cur])
                    if cur < 0:
                        break
                    path.append(cur)
                p = path
        if p:
            for u, v in zip(p[:-1], p[1:]):
                edges.add((min(u, v), max(u, v)))
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1

    # 4. 建父指针 + 每条终点路径
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
    root = start_term
    parent = {root: -1}
    st = [root]
    while st:
        u = st.pop()
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u; st.append(v)

    def walk(n):
        p = [n]
        while parent.get(n, -1) != -1:
            n = parent[n]; p.append(n)
        return p[::-1]

    paths = {}
    for i, e in enumerate(ends):
        p = walk(end_terms[i])
        if len(p) > 1 and p[0] == root:
            paths[e['id']] = p

    # 5. 卡扣主动插入（终点先行：卡扣不在骨架，替换最近线束段 P→C→Q 穿行）
    for j, c in enumerate(clips):
        cn = clip_terms[j]
        if any(cn in p for p in paths.values()):
            continue
        best_eid, best_i, best_d = None, None, np.inf
        for eid, p in paths.items():
            ds = np.linalg.norm(rm.all()[p] - c['pos'], axis=1)
            i = int(np.argmin(ds))
            if ds[i] < best_d:
                best_d, best_i, best_eid = ds[i], i, eid
        if best_eid is not None:
            p = paths[best_eid]; i = best_i
            nxt = i + 1 if i + 1 < len(p) else i
            pa, pb = rm.path(p[i], cn), rm.path(cn, p[nxt])
            if pa is not None and pb is not None:
                paths[best_eid] = p[:i + 1] + pa[1:] + pb[1:] + p[nxt + 1:]

    # 5.5 分叉点定位：分叉点迎合卡扣/终点/其他分叉移动，满足 l1/l2/l5/x 硬约束
    _fp = read_fork_params(case)
    paths = _position_forks(rm, paths, nodes, all_terminals, n_end, _fp)

    # 6. 从最终线束路径重建树，得到与路径一致的分叉点
    final_edges = set()
    for p in paths.values():
        for u, v in zip(p[:-1], p[1:]):
            final_edges.add((min(u, v), max(u, v)))
    final_adj = {}
    for u, v in final_edges:
        final_adj.setdefault(u, []).append(v)
        final_adj.setdefault(v, []).append(u)
    junctions = {n for n in final_adj if len(final_adj[n]) != 2 and n not in all_terminals}
    junctions |= {cur for path in paths.values()
                  for prev, cur, nxt in zip(path, path[1:], path[2:])
                  if prev == nxt and cur not in all_terminals}
    # 从最终线束路径重建父指针
    final_parent = {nodes[0]: -1}
    st = [nodes[0]]
    while st:
        u = st.pop()
        for v in final_adj.get(u, []):
            if v not in final_parent:
                final_parent[v] = u
                st.append(v)
    return rm, paths, nodes, final_parent, junctions


def select_spaced_clips(case, topology, covered, minimum_distance=60.0):
    """Select a maximum-size compatible set using final network arc lengths.

    Distances may cross junctions. Input control_points are not candidates.
    The optimum is for this measured network, not all possible replanned trees.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    request = case['input_request']
    supplied = request.get('required_clips') or request.get('candidate_clips') or []
    candidate_ids = {clip['clip_id'] for clip in supplied}
    candidates = [(clip, exported) for clip, exported, used in zip(
        case['clips'], topology['clip_points'], covered)
        if used and clip['id'] in candidate_ids]
    weights = {}
    for segment in topology['segments']:
        a, b = segment['from_cp_id'], segment['to_cp_id']
        points = np.asarray(topology['edge_paths'][str(segment['id'])]['path_mm'], float)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for edge in ((a, b), (b, a)):
            weights[edge] = min(weights.get(edge, np.inf), length)
    if len(candidates) < 2:
        return [], None
    rows = [edge[0] for edge in weights]
    columns = [edge[1] for edge in weights]
    graph = csr_matrix((list(weights.values()), (rows, columns)),
                       shape=(len(topology['control_points']),) * 2)
    nodes = [exported['control_point_id'] for _, exported in candidates]
    distances = dijkstra(graph, directed=False, indices=nodes)[:, nodes]
    # 相邻卡扣对：同一 harness 路径上连续的卡扣（中间没有其他卡扣才算相邻）
    clip_index = {cp: i for i, cp in enumerate(nodes)}
    adjacent = {}
    for h in topology['harnesses']:
        last_clip = None
        for cp in h['cp_sequence']:
            if cp not in clip_index:
                continue
            if last_clip is not None:
                i, j = clip_index[last_clip], clip_index[cp]
                a, b = (i, j) if i < j else (j, i)
                d = float(distances[a, b])
                adjacent[(a, b)] = min(adjacent.get((a, b), np.inf), d)
            last_clip = cp
    pairs = [(i, j, d) for (i, j), d in adjacent.items() if np.isfinite(d)]
    nearest = min((d for _, _, d in pairs), default=None)
    conflicts = [(i, j, d) for i, j, d in pairs if d < minimum_distance]
    if not conflicts:
        return [], nearest

    count = len(candidates)
    matrix = np.zeros((len(conflicts), count))
    for row, (i, j, _) in enumerate(conflicts):
        matrix[row, [i, j]] = 1
    # A small deterministic preference for input order cannot outweigh one clip.
    priority = 1.0 + (count - np.arange(count)) / (4.0 * count * count)
    solution = milp(-priority, integrality=np.ones(count), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, -np.inf, 1))
    if not solution.success:
        raise RuntimeError('Clip spacing selection failed: ' + solution.message)
    selected = solution.x >= 0.5
    excluded = []
    for index, (clip, _) in enumerate(candidates):
        if selected[index]:
            continue
        blockers = [dict(clip_id=candidates[other][0]['id'], distance_mm=distance)
                    for i, j, distance in conflicts for other in [j if i == index else i]
                    if index in (i, j) and selected[other]]
        excluded.append(dict(clip_id=clip['id'], reason='NETWORK_DISTANCE_BELOW_MINIMUM',
                             minimum_distance_mm=minimum_distance, conflicts=blockers,
                             input_position_mm=clip['pos'].tolist(), input_direction=clip['dir'].tolist()))
    return excluded, nearest
