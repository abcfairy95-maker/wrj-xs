"""端点和途径点解析：起点/终点/控制点（位置 + 方向 + 顺序）。"""
import json
import numpy as np


class Node:
    """一个路径节点（起点/终点/控制点），含位置和方向。"""

    def __init__(self, node_id, pos, direction, kind, order=None):
        self.id = node_id
        self.pos = np.asarray(pos, float)
        d = np.asarray(direction, float)
        n = np.linalg.norm(d)
        self.dir = d / n if n > 1e-9 else d
        self.kind = kind          # 'start' / 'end' / 'control'
        self.order = order

    def __repr__(self):
        return 'Node(%s,%s,pos=%s)' % (self.kind, self.id, np.round(self.pos, 1))


class Endpoints:
    """解析 endpoints（starts/ends）+ control_points。"""

    def __init__(self, request):
        raw = request.get('endpoints', {})
        starts = raw.get('starts', [])
        ends = raw.get('ends', [])
        controls = sorted(request.get('control_points', []), key=lambda n: n.get('order', 0))

        self.start = Node(starts[0]['id'], starts[0]['pos'], starts[0]['dir'], 'start') if starts else None
        self.ends = [Node(e['id'], e['pos'], e['dir'], 'end') for e in ends]
        self.controls = [Node(c['point_id'], c['pos'], c['dir'], 'control', c.get('order'))
                         for c in controls]

        # 链式节点序列：start + controls + ends（control_point 模式）
        self.chain = ([self.start] + self.controls + self.ends) if self.controls else None

    def connections(self):
        """返回连接列表 [(起点节点, 终点节点)]。

        one_click（星型）：START -> 每个 END。
        control_point（链式）：START -> CP1 -> CP2 -> ... -> END。
        """
        if self.chain is not None:
            return list(zip(self.chain[:-1], self.chain[1:]))
        return [(self.start, e) for e in self.ends]

    @property
    def n_endpoints(self):
        return len(self.ends)

    @property
    def n_controls(self):
        return len(self.controls)


def load_endpoints(case_dir):
    with open(case_dir / 'input.json', encoding='utf-8-sig') as f:
        request = json.load(f)
    return Endpoints(request)
