"""结果层：把路径/拓扑组装成完整 result.json（control_points/segments/harnesses/clip_points/edge_paths/metadata）。

CPU/GPU 共享，两个通道都输出完整格式，对齐 result.json 手册 / formal 参考。
"""
import time

import numpy as np

from .data_layer import params, unit


def build_topology(case, rm, paths, nodes, parent, harness_paths, metrics, unreached,
                   col=None, clearance=None, path_samples=None):
    n_end = len(case['ends'])
    edges = set()
    for p in paths.values():
        for u, v in zip(p[:-1], p[1:]):
            edges.add((min(u, v), max(u, v)))
    deg, adj = {}, {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
        adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
    term = set(nodes)
    important = {n for n, d in deg.items() if d != 2} | term
    important |= {cur for path in paths.values()
                  for prev, cur, nxt in zip(path, path[1:], path[2:])
                  if prev == nxt}
    cp_nodes = sorted(important, key=lambda n: tuple(rm.pos(n)))
    cp_id = {n: i for i, n in enumerate(cp_nodes)}
    pos_of = {n: rm.pos(n) for n in cp_nodes}
    kind, tang = {}, {}
    for i, n in enumerate(nodes):
        if i == 0:
            kind[n], tang[n] = 'start', case['start']['dir']
        elif i < 1 + n_end:
            kind[n], tang[n] = 'end', -case['ends'][i - 1]['dir']
        else:
            kind[n], tang[n] = 'clip', case['clips'][i - 1 - n_end]['dir']
    for n in cp_nodes:
        kind.setdefault(n, 'junction'); tang.setdefault(n, np.zeros(3))
    # 卡扣切线符号：对齐从父到子的流动方向（clip.dir 是起点→终点定向）
    for j, c in enumerate(case['clips']):
        n = nodes[1 + n_end + j]
        par = parent.get(n, -1)
        child = next((nb for nb in adj.get(n, []) if nb != par), None)
        if child is not None and par != -1:
            flow = np.asarray(rm.pos(child)) - np.asarray(rm.pos(par))
            if np.linalg.norm(flow) > 1e-6 and np.dot(flow, c['dir']) < 0:
                tang[n] = -np.asarray(tang[n])

    segs, visited = [], set()
    for s in cp_nodes:
        for nb in adj.get(s, []):
            if (min(s, nb), max(s, nb)) in visited:
                continue
            poly = [s]; prev, cur = s, nb
            while cur not in important:
                poly.append(cur)
                nxt = [x for x in adj.get(cur, []) if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            poly.append(cur)
            for a, b in zip(poly[:-1], poly[1:]):
                visited.add((min(a, b), max(a, b)))
            coords = np.array([rm.pos(n) for n in poly])
            segs.append(dict(f=poly[0], t=poly[-1], poly=poly, coords=coords,
                             length=float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())))

    seg_of = {}
    for si, s in enumerate(segs):
        for a, b in zip(s['poly'][:-1], s['poly'][1:]):
            seg_of[(min(a, b), max(a, b))] = si

    cps = []
    for n in cp_nodes:
        tgt = tang[n]
        cps.append(dict(id=cp_id[n], position_mm=[float(x) for x in pos_of[n]],
                        connected_segment_ids=[], degree=deg.get(n, 0),
                        type='end' if kind[n] == 'end' else 'start' if kind[n] == 'start'
                        else 'clip' if kind[n] == 'clip' else 'branch',
                        tangent=None if not np.any(tgt) else [round(float(x), 6) for x in tgt]))

    # 线束（先建，以便统计每段经过的线束数 → 主干/分支）
    harnesses = []
    for i, (eid, p) in enumerate(paths.items()):
        seq = []
        for u, v in zip(p[:-1], p[1:]):
            si = seg_of.get((min(u, v), max(u, v)))
            if si is not None and u in important:
                seq.append(si)
        cps_seq = [cp_id[n] for n in p if n in cp_id]
        harnesses.append(dict(id='H-%03d' % (i + 1), endpoint_id=eid,
                              start_cp_id=cps_seq[0], end_cp_id=cps_seq[-1],
                              segment_sequence=seq, cp_sequence=cps_seq,
                              total_length_mm=round(metrics[eid]['length'], 6), diameter_mm=None))
    n_pass = [0] * len(segs)
    for h in harnesses:
        for si in set(h['segment_sequence']):
            n_pass[si] += 1
    is_trunk = [n_pass[si] >= len(paths) for si in range(len(segs))]

    pref = params(case['prefs'])
    D_t, d_b, r = pref['D_t'], pref['d_b'], pref['r']
    seg_out = []
    for si, s in enumerate(segs):
        a, b = pos_of[s['f']], pos_of[s['t']]
        d = unit(np.asarray(b) - np.asarray(a))
        tf = tang.get(s['f'], np.zeros(3)); tt = tang.get(s['t'], np.zeros(3))
        from_t = tf if np.any(tf) else d
        # to_tangent 约定：卡扣边界用 T_clip（正方向）；非卡扣边界指向段内（反向，即 -旅行方向）
        if kind.get(s['t']) == 'clip':
            to_t = tt if np.any(tt) else d
        else:
            to_t = -tt if np.any(tt) else -d
        D_seg = D_t if is_trunk[si] else d_b
        radius = D_seg / 2.0
        R_min = r * D_seg
        seg_out.append(dict(
            id=si, from_cp_id=cp_id[s['f']], to_cp_id=cp_id[s['t']], mid_cp_ids=[],
            length_mm=round(s['length'], 6),
            from_tangent=[round(float(x), 6) for x in from_t],
            to_tangent=[round(float(x), 6) for x in to_t],
            bend_radius_factor=r, bend_radius_degraded=False,
            actual_min_bend_radius_mm=round(R_min, 4),
            source_edge_ids=[str(si)], source_routing_edge_id=si,
            control_fit_status='ok', control_fit_max_deviation_mm=None,
            N_SEG=1, n_harnesses_passing=n_pass[si],
            D_seg=D_seg, radius_mm=radius, safe_dist_mm=radius + float(case['prefs'].get('clearance_mm', 6)),
            requested_R_min_mm=R_min, effective_R_min_mm=R_min, R_min_mm=R_min,
            is_trunk=is_trunk[si], flow_source='final_harnesses', from_tangent_scale=0.75))
    for si, s in enumerate(segs):
        cps[cp_id[s['f']]]['connected_segment_ids'].append(si)
        cps[cp_id[s['t']]]['connected_segment_ids'].append(si)

    clip_points = []
    for j, c in enumerate(case['clips']):
        n = nodes[1 + n_end + j]
        cpid = cp_id.get(n)
        # 卡扣所在段（卡扣可为段的 from_cp 或 to_cp）
        on_seg = next((s['id'] for s in seg_out
                       if s['from_cp_id'] == cpid or s['to_cp_id'] == cpid), None)
        seg_len = seg_out[on_seg]['length_mm'] if on_seg is not None else None
        seg_radius = seg_out[on_seg]['radius_mm'] if on_seg is not None else None
        # 安装方向：取切向的一个垂向（近似，输入未提供真实安装方向）
        ttang = unit(np.asarray(tang[n], float))
        ref = np.array([0.0, 0.0, 1.0]) if abs(ttang[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        mdir = unit(np.cross(ttang, ref))
        clip_points.append(dict(
            id=j, control_point_id=cpid,
            position_mm=[float(x) for x in c['pos']],
            tangent=[round(float(x), 6) for x in ttang],
            direction=[round(float(x), 6) for x in mdir],
            in_preferred_surface=True, surface_distance_mm=None,
            on_segment_id=on_seg, on_edge_id=on_seg,
            harness_radius_mm=seg_radius, local_arc_length_mm=0.0,
            global_arc_length_mm=None, dist_to_nearest_split_mm=None,
            segment_length_mm=seg_len, status='valid', constraint_status='pass',
            constraint_violations=[], dist_to_nearest_endpoint_mm=None, chain_id=None,
            clip_type='shape', source='user_input', routing_direction=None,
            segment_mean_dev_mm=None, segment_max_dev_mm=None, segment_area_cost_mm2=None,
            invalid_reason=None, distance_to_prev_clip_mm=None, distance_to_next_clip_mm=None))

    harness_paths_out = {'H-%03d' % (i + 1): harness_paths[eid] for i, eid in enumerate(paths)}
    # Preserve the occurrence index through smoothing and slice the final route.
    edge_paths = {}
    for h in harnesses:
        endpoint = h['endpoint_id']
        path = paths[endpoint]
        geometry = np.asarray(harness_paths[endpoint], float)
        if path_samples is not None:
            offsets = path_samples[endpoint]
        else:
            # Legacy direct callers supply sampled geometry without node offsets.
            offsets = []
            cursor = 0
            for node in path:
                cursor += int(np.argmin(np.linalg.norm(geometry[cursor:] - rm.pos(node), axis=1)))
                offsets.append(cursor)
        boundaries = [i for i, node in enumerate(path) if node in important]
        for first, last in zip(boundaries, boundaries[1:]):
            segment_id = seg_of[(min(path[first], path[first + 1]), max(path[first], path[first + 1]))]
            piece = geometry[offsets[first]:offsets[last] + 1]
            if path[first] != segs[segment_id]['f']:
                piece = piece[::-1]
            key = str(segment_id)
            if key in edge_paths:
                published = np.asarray(edge_paths[key]['path_mm'])
                if published.shape != piece.shape or not np.allclose(published, piece, atol=1e-6, rtol=0):
                    raise ValueError('Shared segment %s has inconsistent whole-route geometry' % key)
            else:
                edge_paths[key] = dict(path_mm=piece.tolist())
    edge_to_seg = {str(si): si for si in range(len(segs))}
    metadata = dict(
        version='1.0',
        coordinate_system='车身坐标系（与STL坐标系一致，Z轴向上）',
        units='mm', voxel_size_mm=float(rm.vs),
        origin_mm=[round(float(x), 1) for x in rm.origin],
        n_segments=len(seg_out), n_edge_paths=len(edge_paths),
        n_harnesses=len(harnesses), n_clip_points=len(clip_points),
        n_control_points=len(cps),
        tangent_convention=dict(
            from_tangent='At a clip boundary, T_clip in Harness start-to-end direction; otherwise points from from_cp into the segment.',
            to_tangent='At a clip boundary, T_clip in Harness start-to-end direction and used without negation; otherwise points from to_cp back into the segment.',
            cp_tangent='Local travel direction of this control point on its corresponding edge_path.',
            hermite_usage='At clip boundaries use T_clip directly; otherwise use from_tangent at the start and -to_tangent at the end.'),
        artifact_contract=dict(schema_version='gen_paths_v2_result.v1',
                               run_id='route_%d' % int(time.time())),
        final_unreachable=dict(
            status='unreachable' if unreached else 'ok',
            final_unreachable_count=len(unreached),
            final_unreachable_demands=[u['endpoint_id'] for u in unreached],
            final_unreachable_points=[],
            ends=[u['endpoint_id'] for u in unreached],
            broken_harnesses=[],
            report_path='final_unreachable_report.json' if unreached else None),
        control_point_postprocess=dict(
            enabled=True, status='ok',
            segments_total=len(seg_out), segments_ok=len(seg_out),
            segments_degraded=0, max_deviation_mm=0.0, diagnostics=[]),
        dispatch=dict(schema_version='1.0', source='harness_plan',
                      mode=case['mode'], dag_id='v2',
                      part_category=case['part_category'], algorithm='v2',
                      dag=None, config=None, config_sha256=None,
                      dag_sha256=None, git_commit=None))
    return dict(control_points=cps, segments=seg_out, harnesses=harnesses,
                clip_points=clip_points, edge_paths=edge_paths,
                edge_to_segment_map=edge_to_seg, metadata=metadata), harness_paths_out


def format_result(result, case):
    """Export formal field structures using this run's data; unknown checks stay explicit."""
    request = case['input_request']
    topology = result['topology']
    metadata = topology['metadata']
    control_mode = result['mode'] == 'control_point'
    controls = request.get('control_points', [])
    endpoints = request['endpoints']
    inputs = list(endpoints.get('starts', [])) + list(endpoints.get('ends', []))
    clips = request.get('required_clips') or request.get('candidate_clips') or []
    inputs += clips + controls
    source_by_position = {tuple(item['pos']): item for item in inputs}
    control_ids = {item.get('point_id', item.get('id')) for item in controls}
    cp_by_id = {point['id']: point for point in topology['control_points']}

    if control_mode:
        for point in topology['control_points']:
            # Exported positions are rounded to six decimals.
            source = next((item for position, item in source_by_position.items()
                           if np.linalg.norm(np.asarray(position) - point['position_mm']) <= 1e-4), None)
            source = source or {}
            point_id = source.get('point_id', source.get('clip_id', source.get('id')))
            point.update(point_id=point_id, input_position_mm=source.get('pos'),
                         input_direction=source.get('dir'), order=source.get('order'),
                         is_existing_clip=source.get('is_existing_clip', False))
            if point_id in control_ids:
                point['type'] = 'control'
        topology['user_control_points'] = [point.copy() for point in topology['control_points']
                                           if point.get('point_id') in control_ids]

    for clip in topology['clip_points']:
        point = cp_by_id[clip['control_point_id']]
        if control_mode:
            for key in ('point_id', 'input_position_mm', 'input_direction', 'order', 'is_existing_clip'):
                clip[key] = point[key]
            if point.get('point_id') in control_ids:
                clip['source'] = 'user_control_point'
        clip['in_preferred_surface'] = None

    # A required radius is not a measured radius, and dense path output is not a fit certificate.
    for segment in topology['segments']:
        segment['actual_min_bend_radius_mm'] = None
        segment['control_fit_status'] = 'unknown'
        geometry = np.asarray(topology['edge_paths'][str(segment['id'])]['path_mm'])
        segment['length_mm'] = round(float(np.linalg.norm(np.diff(geometry, axis=0), axis=1).sum()), 6)
        if control_mode:
            segment['routing_span_id'] = None
            segment.pop('from_tangent_scale', None)
    for clip in topology['clip_points']:
        segment_id = clip['on_segment_id']
        if segment_id is not None:
            segment = next(item for item in topology['segments'] if item['id'] == segment_id)
            clip['segment_length_mm'] = segment['length_mm']
            clip['local_arc_length_mm'] = (0.0 if clip['control_point_id'] == segment['from_cp_id']
                                          else segment['length_mm'])

    violations = len(topology['clip_constraint_issues'])
    coverage = topology['clip_constraint_summary']
    metadata['clip_coverage'] = dict(used=coverage['used'], total=coverage['total'])
    topology['clip_constraint_summary'] = dict(status='violation' if violations else 'unknown',
                                               violation_count=violations)
    topology['clip_support_summary'] = dict(status='UNKNOWN', violation_count=0,
                                            primary_status='UNKNOWN', layout_status='not_evaluated',
                                            relaxed_violation_count=0)
    topology['clip_route_spacing'] = dict(
        status='UNKNOWN', original_status='UNKNOWN', effective_status='UNKNOWN', records=[],
        input_errors=[], relaxed_records=[], relaxed_violation_count=0, unresolved_count=0,
        local_permission_regions=[], shape_count=len(topology['clip_points']), supplement_count=0,
        pairing=None, distance_policy=None, global_optimality_claim=False, primary_status='UNKNOWN',
        maximum_gap_fail_count=0, endpoint_coverage_fail_count=0, layout_status='not_evaluated',
        geometry_source='delivered_geometry', published_distance_status='UNKNOWN',
        source_reference_status='UNKNOWN', source_reference_primary_status='UNKNOWN')
    metadata['control_point_postprocess'] = dict(
        enabled=False, status='unknown', segments_total=len(topology['segments']), segments_ok=0,
        segments_degraded=0, max_deviation_mm=None, diagnostics=[])
    metadata['task_id'] = case['task_id']
    metadata['dispatch']['part_category'] = result.pop('part_category', case['part_category'])
    warning = 'Bend radius, surface placement, clip spacing and CAD curve fitting were not evaluated.'
    if warning not in result['warnings']:
        result['warnings'].append(warning)

    if control_mode:
        ordered = sorted(controls, key=lambda item: item.get('order', 0))
        trunk = [endpoints['starts'][0]] + ordered
        pairs = list(zip(trunk, trunk[1:])) + [(trunk[-1], end) for end in endpoints.get('ends', [])]
        spans = [dict(span_id='SPAN-%03d' % (index + 1),
                      from_point_id=start.get('point_id', start.get('id')),
                      to_point_id=end.get('point_id', end.get('id')),
                      status='unknown', reason='ORDERED_SPAN_NOT_EVALUATED', failed_point_ids=[],
                      fallback_records=[], wall_time_s=None, segment_ids=[])
                 for index, (start, end) in enumerate(pairs)]
        result['routing_spans'] = spans
        topology['routing_spans'] = spans
        result['statistics'].update(requested_control_points=len(controls), requested_spans=len(spans),
                                    successful_spans=0, failed_spans=0)
        for harness in topology['harnesses']:
            harness['is_fragment'] = False
        metadata['artifact_contract']['schema_version'] = 'cpr_v2_result.v1'
        metadata.update(cpr_config={}, features_config={}, postprocess_config={}, joint_layout_report={})
        metadata['control_point_postprocess'] = dict(diagnostics=[], max_fit_error_mm=None)
        result['quality_scope'] = 'delivered_geometry'
        result['quality_status'] = 'unknown'
        result.pop('quality_messages', None)
        result['warnings'].append('Ordered CPR spans were not evaluated; routing_spans contain unknown status.')

    for issue in result['issues']:
        issue.setdefault('message_zh', issue['message'])
        issue.setdefault('targets', [dict(type='clip', id=identifier, role='primary')
                                     for identifier in issue.get('target_ids', [])])
        issue.setdefault('details', dict(failure_level='routing'))
        issue.setdefault('diagnostic_id', 'diag-' + issue['code'].lower() + '-' +
                         '-'.join(str(identifier) for identifier in issue.get('target_ids', [])))
    result['issue'] = result['issues'][0] if result['issues'] else None
    return result
