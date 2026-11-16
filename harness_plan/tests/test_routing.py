"""红绿测试：验证核心功能行为（终点全达 + 卡扣全经过）。"""
import os
import json

import numpy as np

from route import route


def _find(pattern):
    for e1 in os.listdir('D:/'):
        p1 = os.path.join('D:/', e1)
        if not os.path.isdir(p1):
            continue
        try:
            subs = os.listdir(p1)
        except PermissionError:
            continue
        for e2 in subs:
            p2 = os.path.join(p1, e2)
            if not os.path.isdir(p2):
                continue
            try:
                subs2 = os.listdir(p2)
            except PermissionError:
                continue
            for e3 in subs2:
                if e3.startswith(pattern) and os.path.isdir(os.path.join(p2, e3)):
                    return os.path.join(p2, e3)
    return None


def _covered(hp, pos, tol=5.0):
    return min(min(np.linalg.norm(np.asarray(p) - np.asarray(pos), axis=1)) for p in hp.values()) < tol


def _run(case_name, device):
    case_dir = _find(case_name)
    assert case_dir is not None, '找不到案例 %s' % case_name
    return route(case_dir, device=device, voxel=30)


def test_endpoints_reachable_cpu():
    res = _run('C07_22ends', 'cpu')
    assert res['statistics']['reached_endpoints'] == res['statistics']['requested_endpoints']


def test_clips_passed_cpu():
    case_dir = _find('C07_22ends')
    req = json.load(open(os.path.join(case_dir, 'input', 'input.json'), encoding='utf-8'))
    clips = req.get('required_clips') or req.get('candidate_clips') or []
    res = _run('C07_22ends', 'cpu')
    for c in clips:
        assert _covered(res['harness_paths'], c['pos']), '卡扣 %s 未经过' % c['clip_id']


def test_gpu_same_coverage():
    """GPU 通道与 CPU 通道同样达成终点全达 + 卡扣全经过。"""
    case_dir = _find('C07_22ends')
    req = json.load(open(os.path.join(case_dir, 'input', 'input.json'), encoding='utf-8'))
    clips = req.get('required_clips') or req.get('candidate_clips') or []
    res = route(case_dir, device='gpu', voxel=30)
    assert res['statistics']['reached_endpoints'] == res['statistics']['requested_endpoints']
    for c in clips:
        assert _covered(res['harness_paths'], c['pos']), 'GPU 卡扣 %s 未经过' % c['clip_id']


def test_b07_control_points_passed():
    case_dir = _find('B07_2ends')
    req = json.load(open(os.path.join(case_dir, 'input', 'input.json'), encoding='utf-8'))
    cps = req.get('control_points', [])
    res = route(case_dir, device='cpu', voxel=30)
    assert res['statistics']['reached_endpoints'] == res['statistics']['requested_endpoints']
    for cp in cps:
        assert _covered(res['harness_paths'], cp['pos']), '控制点 %s 未经过' % cp['point_id']


if __name__ == '__main__':
    test_endpoints_reachable_cpu()
    test_clips_passed_cpu()
    test_gpu_same_coverage()
    test_b07_control_points_passed()
    print('4 项红绿测试全部通过')

