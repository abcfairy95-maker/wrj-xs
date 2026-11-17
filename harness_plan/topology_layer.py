"""拓扑层：MST 主干 + 终点挂叶子 + 卡扣强穿（CPU/GPU 共享，图逻辑不适合 GPU）。"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree


def _slide_forks_away_from_clips(rm, paths, nodes, all_terminals, n_end, l5):
    """把离卡扣 < l5 的分叉点合并到更远的邻居，保证卡扣到分叉 ≥ l5。

    分叉点（degree>=3）若离某卡扣弧长/欧氏距离 < l5，就把它「收缩」到
    离卡扣最远的那个邻居（分叉点移开，卡扣保持 degree=2 穿行）。
    """
    from collections import Counter
    clip_nodes = set(nodes[1 + n_end:])
    for _ in range(30):
        edges = set()
        for p in paths.values():
            for u, v in zip(p[:-1], p[1:]):
                edges.add((min(u, v), max(u, v)))
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        forks = [n for n, nb in adj.items() if len(nb) >= 3 and n not in all_terminals]
        moved = False
        for F in forks:
            d_f = min(float(np.linalg.norm(rm.pos(F) - rm.pos(cn))) for cn in clip_nodes)
            if d_f >= l5:
                continue
            nbs = sorted(adj[F], key=lambda n: -min(float(np.linalg.norm(rm.pos(n) - rm.pos(cn)))
                                                    for cn in clip_nodes))
            N = nbs[0]  # 离卡扣最远的邻居
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
            break
        if not moved:
            break
    return paths


def steiner(rm, case):
    """骨架由终点决定（MST 在 start+ends 上），卡扣退成 degree-2 waypoint 强制穿行。

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

    # 2. MST 主干 = {start + clips}：卡扣天然在主干上（= 穿行），分叉落在卡扣的体素上
    #    终点作为叶子挂接（不参与 MST，避免被当穿行点）
    trunk_nodes = [start_term] + clip_terms
    D = rm.pair(trunk_nodes)
    mst = minimum_spanning_tree(csr_matrix(np.where(np.isfinite(D), D, 0.0))).tocoo()
    edges = set()
    for a, b in zip(mst.row, mst.col):
        p = rm.path(trunk_nodes[a], trunk_nodes[b])
        if p:
            for u, v in zip(p[:-1], p[1:]):
                edges.add((min(u, v), max(u, v)))
    # 终点挂叶子：接到最近的非终端树节点（避免挂到卡扣/终点上，保持卡扣 degree=2）
    tree_nodes = {start_term}
    for u, v in edges:
        tree_nodes.add(u); tree_nodes.add(v)
    for en in end_terms:
        dist, pred = dijkstra(rm.graph(), indices=[en], return_predecessors=True)
        dist = dist[0]; pred = pred[0]
        best, best_d = None, np.inf
        for tn in tree_nodes:
            if tn in all_terminals:
                continue
            if dist[tn] < best_d:
                best_d, best = dist[tn], tn
        if best is not None and np.isfinite(best_d):
            path = [best]; cur = best
            while cur != en:
                cur = int(pred[cur])
                if cur < 0:
                    break
                path.append(cur)
            for u, v in zip(path[:-1], path[1:]):
                edges.add((min(u, v), max(u, v)))

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

    # 5. 卡扣强穿（三级改道）：保证每个卡扣被某条终点线束真正穿过
    #    一级 干净改道 → 二级 方向感知 → 三级 绕行兜底 → 最终兜底
    for j, c in enumerate(clips):
        cn = clip_terms[j]
        if any(cn in p for p in paths.values()):
            continue
        par = parent.get(cn, -1)
        done = False
        if par != -1:
            for eid, p in list(paths.items()):
                if par not in p:
                    continue
                i = p.index(par)
                for side in (1, -1):
                    if not (0 <= i + side < len(p)):
                        continue
                    seg = rm.path(cn, p[i + side])
                    if seg is None or par in seg:
                        continue
                    paths[eid] = p[:i + 1] + seg + p[i + 2:] if side == 1 \
                        else p[:i - 1] + seg[::-1] + p[i:]
                    done = True
                    break
                if done:
                    break
        if done or any(cn in p for p in paths.values()):
            continue
        # 方向感知改道：沿卡扣切向前方找能接回去的节点，把叶子卡扣变穿行点
        if par != -1:
            d = np.asarray(c['dir'], float)
            cnpos = rm.pos(cn)
            for eid, p in list(paths.items()):
                if par not in p:
                    continue
                i = p.index(par)
                flow = rm.pos(p[min(i + 1, len(p) - 1)]) - cnpos
                d = d if np.dot(flow, d) >= 0 else -d
                for k in range(1, min(len(p) - i, 25)):
                    q = p[i + k]
                    if np.dot(rm.pos(q) - cnpos, d) <= 0:
                        break
                    if q in set(nodes):
                        continue
                    seg = rm.path(cn, q)
                    if seg is not None and par not in seg:
                        paths[eid] = p[:i + 1] + seg + p[i + k + 1:]
                        done = True
                        break
                if done:
                    break
        if done or any(cn in p for p in paths.values()):
            continue
        # 绕行兜底：插到最近线束点之后再回来
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
    # 最终兜底：再扫一遍，未覆盖的强制绕行插入
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

    # 5.5 分叉点滑移：把离卡扣 < l5 的分叉点合并到更远的邻居（保证 l5 卡扣到分叉距离）
    l5 = float(case['prefs'].get('clip_min_distance_to_split_mm', 40))
    paths = _slide_forks_away_from_clips(rm, paths, nodes, all_terminals, n_end, l5)

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
    pairs = [(i, j, float(distances[i, j])) for i in range(len(candidates))
             for j in range(i + 1, len(candidates)) if np.isfinite(distances[i, j])]
    nearest = min((distance for _, _, distance in pairs), default=None)
    conflicts = [(i, j, distance) for i, j, distance in pairs if distance < minimum_distance]
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
