"""线束参数：解析 input.json 的 preferences 并做范围校验。

符号约定（见 docs/parameters.md）：
  d  避障距离        r  最小转弯半径倍数      D_t 主干线径
  d_b 分支线径       l1 末端分支长度          l2 分支点间距
  x  最大分支数量    l3~l6 卡扣间距参数
"""
import json


class Parameters:
    def __init__(self, prefs, mode):
        self.mode = mode
        self._raw = prefs

        # 线径
        self.D_t = float(prefs.get('max_harness_diameter_mm', 12.0))
        self.d_b = float(prefs.get('min_harness_diameter_mm', 6.0))

        # 转弯半径倍数 r（one_click 上限 3，control_point 上限 6）
        self.r = int(prefs.get('bend_radius_factor', 2))

        # 避障距离 d
        self.d = float(prefs.get('clearance_mm', 6.0))

        # 末端分支长度 l1 / 分叉间距 l2 / 分叉度 x
        self.l1 = int(prefs.get('terminal_branch_min_length_mm', 60))
        self.l2 = int(prefs.get('split_point_min_spacing_mm', 75))
        self.x = int(prefs.get('max_fork_degree', 5))
        self.first_fork_min = float(prefs.get('start_to_first_fork_min_distance_mm', 40.0))

        # 卡扣参数 l3~l6
        self.l3 = int(prefs.get('clip_min_spacing_mm', 100))
        self.l4 = int(prefs.get('clip_max_spacing_mm', 150))
        self.l5 = int(prefs.get('clip_min_distance_to_split_mm', 40))
        self.l6 = int(prefs.get('clip_min_distance_to_endpoint_mm', 60))
        self.generate_clips = bool(prefs.get('generate_clips', False))
        self.clip_template_id = prefs.get('clip_template_id', '')

        # 派生量
        self.wire_radius = self.D_t / 2.0               # 线束实体半径
        self.min_bend_radius = self.r * self.D_t        # 最小转弯半径 R = r * D_t
        self.clearance_line = self.wire_radius + self.d # 中心线到障碍净距

        self.warnings = []
        self.validate()

    def validate(self):
        w = self.warnings
        if not (0 < self.d <= 10):
            w.append('d(避障距离) 超出 (0,10]，当前 %s' % self.d)
        if not (0 < self.d_b <= 10):
            w.append('d_b(分支线径) 超出 (0,10]，当前 %s' % self.d_b)
        if not (self.d_b < self.D_t <= 50):
            w.append('D_t(主干线径) 需满足 d_b<D_t<=50，当前 d_b=%s D_t=%s' % (self.d_b, self.D_t))
        r_max = 6 if self.mode == 'control_point' else 3
        if not (0 < self.r <= r_max):
            w.append('r(半径倍数) 超出 (0,%d]，当前 %s' % (r_max, self.r))
        if not (0 < self.l1 <= 100):
            w.append('l1(末端分支长度) 超出 (0,100]，当前 %s' % self.l1)
        if not (0 < self.l2 <= 100):
            w.append('l2(分叉间距) 超出 (0,100]，当前 %s' % self.l2)
        if self.x not in (4, 5):
            w.append('x(分叉度) 仅支持 4 或 5，当前 %s' % self.x)
        if not (60 < self.l3 <= 150):
            w.append('l3(卡扣最小距离) 超出 (60,150]，当前 %s' % self.l3)
        if not (self.l3 < self.l4 <= 300):
            w.append('l4(卡扣最大距离) 需 l3<l4<=300，当前 l3=%s l4=%s' % (self.l3, self.l4))

    def to_dict(self):
        return dict(mode=self.mode, D_t=self.D_t, d_b=self.d_b, r=self.r, d=self.d,
                    l1=self.l1, l2=self.l2, x=self.x, first_fork_min=self.first_fork_min,
                    l3=self.l3, l4=self.l4, l5=self.l5, l6=self.l6,
                    wire_radius=self.wire_radius, min_bend_radius=self.min_bend_radius,
                    clearance_line=self.clearance_line, warnings=self.warnings)


def load_parameters(case_dir, mode=None):
    with open(case_dir / 'input.json', encoding='utf-8-sig') as f:
        request = json.load(f)
    mode = mode or request.get('mode', 'one_click')
    return Parameters(request.get('preferences', {}), mode)
