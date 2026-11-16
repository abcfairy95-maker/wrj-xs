"""最终曲线与拓扑校验；记录不满足项，不把点位覆盖当作合格穿行。"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def validate_result(result, case, roadmap):
    topology = result['topology']
    points = {point['id']: point for point in topology['control_points']}
    segments = {segment['id']: segment for segment in topology['segments']}
    weights = {}
    issues, clip_reports = [], []

    def violation(code, targets, details):
        identifiers = [str(value) for value in targets]
        kind = 'clip' if code.startswith('CLIP_') else ('segment' if code.startswith('BEND_') else 'control_point')
        issues.append(dict(code=code, category='geometry', severity='error',
                           message=code, message_zh=code, target_ids=identifiers,
                           targets=[dict(type=kind, id=value, role='primary') for value in identifiers],
                           diagnostic_id='diag-' + code.lower() + '-' + '-'.join(identifiers), details=details))

    for segment in segments.values():
        curve = np.asarray(topology['edge_paths'][str(segment['id'])]['path_mm'])
        length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        segment['length_mm'] = length
        for edge in [(segment['from_cp_id'], segment['to_cp_id']), (segment['to_cp_id'], segment['from_cp_id'])]:
            weights[edge] = min(weights.get(edge, np.inf), length)
        radius = getattr(roadmap, 'final_segment_radii', {}).get(segment['id'])
        segment['actual_min_bend_radius_mm'] = radius
        if radius is not None and radius + 1e-6 < segment['requested_R_min_mm']:
            violation('BEND_RADIUS_BELOW_MINIMUM', [segment['id']],
                      dict(actual_mm=radius, minimum_mm=segment['requested_R_min_mm']))
    graph = csr_matrix((list(weights.values()), ([edge[0] for edge in weights], [edge[1] for edge in weights])),
                       shape=(len(points), len(points)))
    branches = [node for node, point in points.items() if point['degree'] >= 3]
    terminals = [node for node, point in points.items() if point['type'] in ('start', 'end')]
    customer_ids = {clip['clip_id'] for clip in (case['input_request'].get('required_clips') or case['input_request'].get('candidate_clips') or [])}
    l5 = float(case['prefs'].get('clip_min_distance_to_split_mm', 40))
    l6 = 60.0
    effective = []
    for supplied, exported in zip(case['clips'], topology['clip_points']):
        node = exported['control_point_id']
        errors, angles, sides = [], [], []
        if points[node]['degree'] != 2:
            errors.append('CLIP_NOT_TRANSIT_DEGREE_TWO')
        present = False
        for harness in topology['harnesses']:
            sequence = harness['cp_sequence']
            curve = np.asarray(result['harness_paths'][harness['id']])
            occurrences = [index for index, value in enumerate(sequence) if value == node]
            if len(occurrences) > 1:
                errors.append('CLIP_REPEATED_VISIT')
            for index in occurrences:
                present = True
                if index == 0 or index == len(sequence) - 1 or sequence[index - 1] == sequence[index + 1]:
                    errors.append('CLIP_RETURN_OR_ENDPOINT')
                    continue
                position = np.asarray(supplied['pos'])
                sample = int(np.argmin(np.linalg.norm(curve - position, axis=1)))
                if np.linalg.norm(curve[sample] - position) > 1e-4:
                    errors.append('CLIP_POSITION_MISMATCH')
                    continue
                if not 0 < sample < len(curve) - 1:
                    errors.append('CLIP_RETURN_OR_ENDPOINT')
                    continue
                axis = np.asarray(supplied['dir'])
                before, after = curve[sample - 1] - position, curve[sample + 1] - position
                opposite = float(before @ axis) < -1e-6 and float(after @ axis) > 1e-6
                sides.append(opposite)
                for vector in (-before, after):
                    cosine = float(np.dot(vector, axis) / max(np.linalg.norm(vector), 1e-15))
                    angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
                if not opposite:
                    errors.append('CLIP_NOT_OPPOSITE_SIDES')
                # 两侧保留直线门段，容差仅用于浮点计算，不是放宽输入方向。
                if max(angles[-2:]) > 1e-3:
                    errors.append('CLIP_DIRECTION_MISMATCH')
        if not present:
            errors.append('CLIP_UNREACHED')
        distances = dijkstra(graph, indices=node)
        branch_distance = min((float(distances[b]) for b in branches), default=np.inf)
        terminal_distance = min((float(distances[t]) for t in terminals), default=np.inf)
        customer = supplied['id'] in customer_ids
        if customer and branch_distance < l5 - 1e-6:
            errors.append('CLIP_BRANCH_DISTANCE_BELOW_L5')
        if customer and terminal_distance < l6 - 1e-6:
            errors.append('CLIP_TERMINAL_DISTANCE_BELOW_L6')
        errors = sorted(set(errors))
        exported.update(status='valid' if not errors else 'invalid', constraint_status='pass' if not errors else 'fail',
                        constraint_violations=errors, invalid_reason=','.join(errors) or None,
                        dist_to_nearest_split_mm=None if not np.isfinite(branch_distance) else branch_distance,
                        dist_to_nearest_endpoint_mm=None if not np.isfinite(terminal_distance) else terminal_distance)
        report = dict(clip_id=supplied['id'], customer_clip=customer, degree=points[node]['degree'],
                      valid_transit=not errors, max_direction_error_deg=max(angles, default=None),
                      branch_distance_mm=exported['dist_to_nearest_split_mm'],
                      terminal_distance_mm=exported['dist_to_nearest_endpoint_mm'], violations=errors)
        clip_reports.append(report)
        if customer and not errors:
            effective.append(supplied['id'])
        for error in errors:
            violation(error, [supplied['id']], report)
    if case['mode'] == 'one_click':
        l1 = float(case['prefs'].get('terminal_branch_min_length_mm', 60))
        l2 = float(case['prefs'].get('split_point_min_spacing_mm', 75))
        for node in branches:
            if points[node]['type'] == 'clip':
                violation('BRANCH_AT_CLIP', [node], {})
            if points[node]['degree'] > 4:
                violation('BRANCH_DEGREE_EXCEEDED', [node], dict(actual=points[node]['degree'], maximum=4))
            distances = dijkstra(graph, indices=node)
            for end in terminals:
                if points[end]['type'] == 'end' and distances[end] < l1 - 1e-6:
                    violation('TERMINAL_BRANCH_BELOW_L1', [node, end], dict(actual_mm=float(distances[end]), minimum_mm=l1))
            for other in branches:
                if other > node and distances[other] < l2 - 1e-6:
                    violation('BRANCH_DISTANCE_BELOW_L2', [node, other], dict(actual_mm=float(distances[other]), minimum_mm=l2))
    # l4逐条链检查，补扣暂缓，报告缺口而不伪造安装合格。
    clip_nodes = {clip['control_point_id'] for supplied, clip in zip(case['clips'], topology['clip_points']) if supplied['id'] in customer_ids}
    adjacency = {}
    for (a, b), weight in weights.items():
        adjacency.setdefault(a, []).append((b, weight))
    boundaries = clip_nodes | set(branches) | set(terminals)
    seen, gaps = set(), []
    maximum = float(case['prefs'].get('clip_max_spacing_mm', 150))
    for first in sorted(boundaries):
        for next_node, distance in adjacency.get(first, []):
            edge = tuple(sorted((first, next_node)))
            if edge in seen:
                continue
            seen.add(edge)
            previous, current, total = first, next_node, distance
            while current not in boundaries:
                following = [(node, length) for node, length in adjacency.get(current, []) if node != previous]
                if len(following) != 1:
                    break
                other, length = following[0]
                seen.add(tuple(sorted((current, other))))
                previous, current, total = current, other, total + length
            if first in clip_nodes and current in clip_nodes and total > maximum + 1e-6:
                gaps.append(dict(from_cp=first, to_cp=current, length_mm=total, maximum_mm=maximum))
    metadata = topology['metadata']
    metadata['final_geometry_validation'] = dict(
        clip_transit=clip_reports, valid_customer_clip_ids=effective,
        independent_branch_count=sum(points[node]['type'] not in ('clip', 'start', 'end') for node in branches),
        independent_branches=[dict(control_point_id=node, position_mm=points[node]['position_mm'], degree=points[node]['degree'])
                              for node in branches if points[node]['type'] not in ('clip', 'start', 'end')],
        supplemental_clip_status='deferred_by_user', l4_unfilled_gaps=gaps,
        collision_status='not_certified_vertex_distance_model',
        radius_measurement='analytic_spline_derivatives_at_export_samples',
        violations=issues, planning_issues=getattr(roadmap, 'planning_issues', []))
    metadata['clip_selection']['position_covered_count'] = metadata['clip_selection']['used_count']
    metadata['clip_selection']['used_count'] = len(effective)
    metadata['clip_selection']['used_clip_ids'] = effective
    topology['clip_constraint_summary'] = dict(used=sum(item['valid_transit'] for item in clip_reports), total=len(clip_reports), status='pass' if all(item['valid_transit'] for item in clip_reports) else 'violation',
                                               violation_count=sum(not item['valid_transit'] for item in clip_reports))
    topology['clip_constraint_issues'] = [issue for issue in issues if issue['code'].startswith('CLIP_')]
    result['issues'].extend(issues)
    result['violations'].extend(issues)
    result['warnings'] = [warning for warning in result['warnings'] if not warning.startswith('Bend radius, surface placement')]
    result['warnings'].append('Obstacle clearance and CAD fitting are not certified; automatic supplementary clips deferred by user.')
    result['success'] = False if issues or gaps or result['fallback_records'] else result['success']
    result['status'] = 'partial_success' if not result['success'] else result['status']
    result['quality_status'] = 'violation' if issues else 'unknown'
    result['issue'] = result['issues'][0] if result['issues'] else None
