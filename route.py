"""单入口：route(case, device='cpu'|'gpu')。

四层流水线（CPU/GPU 双通道，共享数据/拓扑/结果层）：
  数据层  data_layer     load_case / params / Collision        （共享）
  地图层  map_layer      Roadmap 体素路图（含 device 分发）
  寻路层  pathfind_layer CPU Dijkstra/A* / GPU 波前传播
  拓扑层  topology_layer steiner：MST + 终点挂叶子 + 卡扣强穿   （共享）
  平滑层  smooth_layer   CPU Hermite / GPU Adam 可微分
  结果层  result_layer   build_topology → 完整 result.json      （共享）

用法：
  python route.py --case <case_dir> [--device cpu|gpu] [--voxel 15] [--out result.json]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from harness_plan.data_layer import load_case, params, Collision, unit, SurfacePreference
from harness_plan.map_layer import Roadmap
from harness_plan.topology_layer import steiner, select_spaced_clips
from harness_plan.smooth_layer import smooth_network
from harness_plan.result_layer import build_topology, format_result


def route(case_dir, voxel=None, input_file=None, out=None, turn_w=30.0,
          search='dijkstra', pref_weight=None, *, device='cpu'):
    if device not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    t0 = time.time()
    case = load_case(case_dir, input_file)
    p = params(case['prefs'])
    col = Collision(case['env'])
    P = np.array([case['start']['pos']] + [e['pos'] for e in case['ends']]
                 + [c['pos'] for c in case['clips']])
    if voxel is None:
        voxel = float(np.clip(round((np.prod(P.max(0) - P.min(0)) / 60000) ** (1 / 3)), 15, 30))

    surface = (SurfacePreference(case['surface_parts'], p['clearance'], voxel / 2)
               if case.get('surface_parts') and pref_weight != 0 else None)
    if pref_weight is None:
        pref_weight = 2.0 if surface is not None else 0.0
    excluded_clips = []
    while True:
        rm = Roadmap(col, P.min(0) - 40, P.max(0) + 40, voxel, p['clearance'], device=device)
        rm.use_astar = (search == 'astar')
        if pref_weight > 0 and surface is not None:
            rm.set_preference(surface, pref_weight)
        rm.grid_edges()
        _, paths, nodes, parent, junctions = steiner(rm, case)

        n_end = len(case['ends'])
        protected_nodes = set(nodes) | junctions
        directions = {nodes[0]: case['start']['dir']}
        directions.update({nodes[1 + i]: end['dir'] for i, end in enumerate(case['ends'])})
        directions.update({nodes[1 + n_end + i]: clip['dir'] for i, clip in enumerate(case['clips'])})
        geometry, path_samples = smooth_network(rm, paths, protected_nodes, directions, device)
        harness_paths = {endpoint: points.tolist() for endpoint, points in geometry.items()}
        metrics = {endpoint: dict(length=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
                   for endpoint, points in geometry.items()}

        unreached = []
        for end in case['ends']:
            geometry = np.asarray(harness_paths.get(end['id'], []), float)
            valid = (len(geometry) > 0 and np.isfinite(geometry).all()
                     and np.linalg.norm(geometry[0] - case['start']['pos']) <= 1e-4
                     and np.linalg.norm(geometry[-1] - end['pos']) <= 1e-4)
            if not valid:
                reason = 'missing_harness' if end['id'] not in paths else 'endpoint_geometry_mismatch'
                unreached.append(dict(endpoint_id=end['id'], reason=reason))
        used_clips = []
        for j, clip in enumerate(case['clips']):
            node = nodes[1 + n_end + j]
            covered = any(node in path and np.min(np.linalg.norm(
                np.asarray(harness_paths[eid]) - clip['pos'], axis=1)) <= 1e-4
                for eid, path in paths.items())
            used_clips.append(covered)
        topo, hp = build_topology(case, rm, paths, nodes, parent, harness_paths, metrics, unreached,
                                  col=col, clearance=p['clearance'], path_samples=path_samples)
        excluded, final_minimum = select_spaced_clips(case, topo, used_clips, minimum_distance=60.0)
        if not excluded:
            break
        excluded_clips.extend(excluded)
        removed_ids = {item['clip_id'] for item in excluded}
        case['clips'] = [clip for clip in case['clips'] if clip['id'] not in removed_ids]

    issues = []
    for u in unreached:
        issues.append(dict(
            code='ENDPOINT_UNREACHABLE', category='connectivity', severity='error',
            message='Endpoint %s did not produce a deliverable route.' % u['endpoint_id'],
            message_zh='终点 %s 未生成可交付路线。' % u['endpoint_id'],
            targets=[dict(type='endpoint', id=u['endpoint_id'], role='primary')],
            details=dict(failure_level='routing'),
            target_ids=[u['endpoint_id']],
            diagnostic_id='diag-unreachable-%s' % u['endpoint_id']))
    clip_issues = []
    for clip, covered in zip(case['clips'], used_clips):
        if not covered:
            clip_issues.append(dict(
                code='CLIP_UNREACHED', category='connectivity', severity='error',
                message='Required clip %s is not covered by final geometry.' % clip['id'],
                target_ids=[clip['id']]))
    issues.extend(clip_issues)
    for clip_output, covered in zip(topo['clip_points'], used_clips):
        if not covered:
            clip_output.update(status='invalid', constraint_status='fail',
                               invalid_reason='unreached', constraint_violations=['CLIP_UNREACHED'])
    status = 'success' if not issues else 'partial_success'
    result = dict(mode=case['mode'], part_category=case['part_category'],
                  topology=dict(metadata=topo['metadata'],
                                control_points=topo['control_points'],
                                segments=topo['segments'], harnesses=topo['harnesses'],
                                clip_points=topo['clip_points'],
                                edge_paths=topo['edge_paths'],
                                edge_to_segment_map=topo['edge_to_segment_map'],
                                clip_constraint_summary=dict(used=sum(used_clips),
                                                             total=len(case['clips'])),
                                clip_constraint_issues=clip_issues,
                                clip_route_spacing={},
                                clip_support_issues=[],
                                clip_support_summary=dict(used=sum(used_clips),
                                                          total=len(case['clips']))),
                  harness_paths=hp,
                  statistics=dict(requested_endpoints=len(case['ends']),
                                  reached_endpoints=len(case['ends']) - len(unreached),
                                  n_segments=len(topo['segments']),
                                  n_harnesses=len(paths),
                                  n_control_points=len(topo['control_points']),
                                  n_clip_points=len(case['clips']),
                                  total_length_mm=round(sum(m['length'] for m in metrics.values()), 6)),
                  fallback_records=[], quality_messages=[], violations=[],
                  unreached_endpoints=unreached,
                  delivery_requirements=dict(generate_clips=bool(case['prefs'].get('generate_clips', False))),
                  status=status,
                  success=not bool(issues),
                  issue=issues[0] if issues else None,
                  issues=issues,
                  error_code=None, error_message=None, warnings=[],
                  _metrics=metrics)
    result.pop('_metrics', None)
    result = format_result(result, case)
    supplied = case['input_request'].get('required_clips') or case['input_request'].get('candidate_clips') or []
    supplied_ids = {clip['clip_id'] for clip in supplied}
    selected_ids = [clip['id'] for clip in case['clips'] if clip['id'] in supplied_ids]
    used_ids = [clip['id'] for clip, used in zip(case['clips'], used_clips)
                if used and clip['id'] in supplied_ids]
    result['topology']['metadata']['clip_selection'] = dict(
        minimum_distance_mm=60.0, distance_metric='shortest_arc_length_on_connected_network',
        input_count=len(supplied), selected_count=len(selected_ids), used_count=len(used_ids),
        selected_clip_ids=selected_ids, used_clip_ids=used_ids, excluded_clips=excluded_clips,
        final_minimum_distance_mm=final_minimum, global_optimality_claim=False,
        tie_break='input_order', scope='connected_customer_clips')
    if out is None:
        out = Path(case_dir).parent / (Path(case_dir).name + '_result_new.json')
    json.dump(result, open(Path(out), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ends=%d/%d clips=%d/%d len=%.0fmm %.1fs (device=%s) -> %s' % (
        result['statistics']['reached_endpoints'], len(case['ends']), sum(used_clips), len(case['clips']),
        result['statistics']['total_length_mm'], time.time() - t0, device, out))
    return result


def _torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    ap.add_argument('--voxel', type=float, default=None)
    ap.add_argument('--input-file', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--search', default='dijkstra', choices=['dijkstra', 'astar'])
    ap.add_argument('--pref', type=float, default=None, help='贴面偏好权重(有贴面默认2.0，0关闭)')
    a = ap.parse_args()
    route(a.case, device=a.device, voxel=a.voxel, input_file=a.input_file, out=a.out,
          search=a.search, pref_weight=a.pref)
