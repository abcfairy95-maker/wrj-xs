"""拓扑层：MST 主干 + 终点挂叶子 + 卡扣强穿（CPU/GPU 共享，图逻辑不适合 GPU）。"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree


def _network_edges(paths):
    return {tuple(sorted((a, b))) for path in paths.values() for a, b in zip(path, path[1:])}


def _adjacency(edges):
    result = {}
    for a, b in edges:
        result.setdefault(a, set()).add(b)
        result.setdefault(b, set()).add(a)
    return result


def _rooted_tree(edges, root):
    adjacency = _adjacency(edges)
    parent, stack = {root: -1}, [root]
    while stack:
        node = stack.pop()
        for other in sorted(adjacency.get(node, ())):
            if other not in parent:
                parent[other] = node
                stack.append(other)
    return adjacency, parent


def _root_path(parent, node):
    if node not in parent:
        return []
    path = [node]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    return path[::-1]


def _tree_chains(edges, root, protected):
    adjacency, parent = _rooted_tree(edges, root)
    important = protected | {node for node, neighbours in adjacency.items() if len(neighbours) != 2}
    chains = []
    for first in sorted(important & parent.keys()):
        for next_node in sorted(adjacency.get(first, ())):
            if parent.get(next_node) != first:
                continue
            chain = [first, next_node]
            while chain[-1] not in important:
                following = next(node for node in adjacency[chain[-1]] if node != chain[-2])
                chain.append(following)
            chains.append(chain)
    return chains


def _insert_transit(rm, edges, root, center, clip, protected):
    """替换一条无分叉链，接入卡扣两侧；新连接不得再次碰到保留树。

    候选按到卡扣的几何距离确定性排序。两个搜索顺序用于避免先铺的一侧
    挡住另一侧；未找到只表示此次有限候选搜索失败，不声称物理不可达。
    """
    chains = _tree_chains(edges, root, protected)
    chains.sort(key=lambda chain: (min(np.linalg.norm(rm.pos(node) - clip['pos']) for node in chain), chain[0]))
    half = min(rm.vs / 2, float(rm.routing_params['l5']) / 2)
    direction = np.asarray(clip['dir'])
    entry_pos, exit_pos = clip['pos'] - half * direction, clip['pos'] + half * direction
    if not (rm.collision.free_seg(entry_pos, clip['pos'], rm.clearance)
            and rm.collision.free_seg(clip['pos'], exit_pos, rm.clearance)):
        return None, 'CLIP_PASSAGE_CLEARANCE'
    entry, exit_node = rm.append_points([entry_pos, exit_pos])
    for chain in chains:
        chain_edges = {tuple(sorted(pair)) for pair in zip(chain, chain[1:])}
        retained = edges - chain_edges
        retained_nodes = {node for edge in retained for node in edge}
        a, b = chain[0], chain[-1]
        for reverse_search in (False, True):
            if reverse_search:
                right = rm.transit_path(exit_node, [b], retained_nodes - {b})
                left = rm.transit_path(a, [entry], (retained_nodes | set(right or [])) - {a}) if right else None
            else:
                left = rm.transit_path(a, [entry], retained_nodes - {a})
                right = rm.transit_path(exit_node, [b], (retained_nodes | set(left or [])) - {b}) if left else None
            if not left or not right:
                continue
            inserted = left + [center] + right
            if len(set(inserted)) != len(inserted):
                continue
            updated = retained | {tuple(sorted(pair)) for pair in zip(inserted, inserted[1:])}
            adjacency, parent = _rooted_tree(updated, root)
            if len(updated) != len(parent) - 1 or len(adjacency[center]) != 2:
                continue
            ordered = [[rm.control_order[node] for node in _root_path(parent, last) if node in rm.control_order]
                       for last in parent if len(adjacency.get(last, ())) == 1]
            if any(sequence != sorted(sequence) for sequence in ordered):
                continue
            rm.transit_gates[center] = (entry, exit_node)
            rm.extra_protected.update([entry, exit_node])
            rm.extra_directions.update({entry: direction, exit_node: direction})
            return updated, None
    return None, 'NO_DISJOINT_TRANSIT_ROUTE_IN_CANDIDATES'


def steiner(rm, case):
    """终点骨架与卡扣穿行分离；保留树的连通性，不允许卡扣叶子/分叉。"""
    endpoints, clips = case['ends'], case['clips']
    positions = [case['start']['pos']] + [end['pos'] for end in endpoints] + [clip['pos'] for clip in clips]
    nodes = rm.append_points(positions)
    root, end_nodes, clip_nodes = nodes[0], nodes[1:1 + len(endpoints)], nodes[1 + len(endpoints):]
    rm.extra_protected, rm.extra_directions, rm.transit_gates = set(), {}, {}
    rm.planning_issues = []
    rm.routing_params = dict(l1=float(case['prefs'].get('terminal_branch_min_length_mm', 60)),
                             l2=float(case['prefs'].get('split_point_min_spacing_mm', 75)),
                             l5=float(case['prefs'].get('clip_min_distance_to_split_mm', 40)))
    controls = {point.get('point_id', point.get('id')) for point in case['input_request'].get('control_points', [])}
    protected = set(nodes)
    terminal_radius = (float(case['prefs'].get('max_harness_diameter_mm', 12)) / 2
                       if case['mode'] == 'control_point' else float(case['prefs'].get('min_harness_diameter_mm', 6)) / 2)
    rm.connector_clearance = {node: terminal_radius + rm.effective_clearance_mm for node in end_nodes}
    rm.control_order = {node: index for index, (node, clip) in enumerate(zip(clip_nodes, clips)) if clip['id'] in controls}
    edges, connected_ends = set(), []

    # 输入终点间的最短路构成MST草稿，只用于确定初始接入顺序，终点不作分叉。
    terminals = [root] + end_nodes
    pair_paths, distances = {}, np.zeros((len(terminals), len(terminals)))
    for i, a in enumerate(terminals):
        for j in range(i + 1, len(terminals)):
            path = rm.transit_path(a, [terminals[j]])
            pair_paths[i, j] = path
            length = rm.route_cost(path) if path else np.inf
            distances[i, j] = distances[j, i] = length
    draft = minimum_spanning_tree(csr_matrix(np.where(np.isfinite(distances), distances, 0))).tocoo()
    draft_adj = _adjacency({(int(a), int(b)) for a, b in zip(draft.row, draft.col)})
    order, seen, stack = [], {0}, [0]
    while stack:
        item = stack.pop(0)
        for other in sorted(draft_adj.get(item, ())):
            if other not in seen:
                seen.add(other); order.append(other - 1); stack.append(other)
    order += [i for i in range(len(endpoints)) if i not in order]
    for index in order:
        end = end_nodes[index]
        if not edges:
            path = pair_paths.get((0, index + 1))
        else:
            adjacency, parent = _rooted_tree(edges, root)
            # 距现有分叉与末端的弧长限制在当前树上判定；不以直线距离误删候选。
            tree_nodes = set(adjacency)
            targets = []
            for candidate in sorted(tree_nodes - protected - getattr(rm, "rejected_branch_nodes", set())):
                if len(adjacency[candidate]) >= 4:
                    continue
                valid = True
                for neighbour in adjacency[candidate]:
                    previous, current, length = candidate, neighbour, float(np.linalg.norm(rm.pos(candidate) - rm.pos(neighbour)))
                    while len(adjacency[current]) == 2 and current not in protected:
                        following = next(n for n in adjacency[current] if n != previous)
                        length += float(np.linalg.norm(rm.pos(current) - rm.pos(following)))
                        previous, current = current, following
                    if current in connected_ends and length < rm.routing_params['l1']:
                        valid = False
                    if len(adjacency[current]) >= 3 and length < rm.routing_params['l2']:
                        valid = False
                if valid:
                    targets.append(candidate)
            branch = None
            while targets:
                branch = rm.transit_path(end, targets, tree_nodes - set(targets))
                if not branch:
                    break
                length = sum(float(np.linalg.norm(rm.pos(a) - rm.pos(b))) for a, b in zip(branch, branch[1:]))
                if length >= rm.routing_params['l1']:
                    break
                targets.remove(branch[-1])
                branch = None
            path = branch[::-1] if branch else None
        if path:
            edges |= {tuple(sorted(pair)) for pair in zip(path, path[1:])}
            connected_ends.append(end)
    # 固定控制点先处理，客户候选按输入顺序试插入；后续间距筛选保留既有职责。
    clip_order = sorted(range(len(clips)), key=lambda index: (clips[index]['id'] not in controls, index))
    for index in clip_order:
        center, clip = clip_nodes[index], clips[index]
        updated, reason = _insert_transit(rm, edges, root, center, clip, protected | rm.extra_protected)
        if updated is None:
            rm.planning_issues.append(dict(clip_id=clip['id'], reason=reason))
        else:
            edges = updated
    adjacency, parent = _rooted_tree(edges, root)
    paths = {end['id']: _root_path(parent, node) for end, node in zip(endpoints, end_nodes) if node in parent}
    junctions = {node for node, neighbours in adjacency.items() if len(neighbours) >= 3 and node not in protected}
    return rm, paths, nodes, parent, junctions


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


def branch_constraint_conflicts(topology, routing_params, clips, customer_ids):
    """按最终中心线弧长识别不合格分叉，卡扣坐标及方向不参与调整。"""
    points = topology['control_points']
    weights = {}
    for segment in topology['segments']:
        curve = np.asarray(topology['edge_paths'][str(segment['id'])]['path_mm'])
        length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        a, b = segment['from_cp_id'], segment['to_cp_id']
        weights[a, b] = weights[b, a] = length
    graph = csr_matrix((list(weights.values()), ([a for a, b in weights], [b for a, b in weights])),
                       shape=(len(points),) * 2)
    branches = [point for point in points if point['degree'] >= 3]
    customer_clips = [(clip['id'], exported['control_point_id']) for clip, exported in zip(clips, topology['clip_points'])
                      if clip['id'] in customer_ids]
    conflicts = []
    for branch in branches:
        distances = dijkstra(graph, indices=branch['id'])
        constraints = [('l1', point['id'], point['id']) for point in points if point['type'] == 'end']
        constraints += [('l2', other['id'], other['id']) for other in branches if other['id'] != branch['id']]
        constraints += [('l5', node, identifier) for identifier, node in customer_clips]
        violations = [dict(constraint=rule, target_id=identifier, actual_mm=float(distances[node]),
                           minimum_mm=routing_params[rule]) for rule, node, identifier in constraints
                      if distances[node] < routing_params[rule] - 1e-6]
        if violations:
            conflicts.append(dict(position_mm=branch['position_mm'], violations=violations))
    return conflicts
