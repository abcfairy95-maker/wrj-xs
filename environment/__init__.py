"""地图构建层：统一加载一个案例的全部输入（参数/端点途径点/三类STL/贴面避让）。"""
from .params import Parameters
from .points import Endpoints
from .meshes import Meshes
from .surfaces import Surfaces


class Environment:
    """一个案例的完整环境（地图构建层输出）。

    包含：
    - params：参数（线径/半径/净空/间距...，含范围校验）
    - endpoints：起点/终点/途径点（位置+方向+顺序）
    - meshes：三类 STL（障碍 env_ / 贴面 sheet_part / 避让 avoid_part）
    - surfaces：贴面区（routing_surfaces）+ 局部避让面（avoidance_surfaces）
    """

    def __init__(self, case_dir):
        import json
        self.case_dir = case_dir
        with open(case_dir / 'input.json', encoding='utf-8-sig') as f:
            self.request = json.load(f)
        self.mode = self.request.get('mode', 'one_click')
        self.part_category = self.request.get('part_category', '')

        self.params = Parameters(self.request.get('preferences', {}), self.mode)
        self.endpoints = Endpoints(self.request)
        self.meshes = Meshes(case_dir / 'input')
        self.surfaces = Surfaces(self.request)

    # 便捷访问
    @property
    def obstacles(self):
        return self.meshes.meshes.get('env')

    @property
    def sheets(self):
        return self.meshes.meshes.get('sheet')

    @property
    def avoids(self):
        return self.meshes.meshes.get('avoid')

    def obstacle_distance(self, points):
        return self.meshes.obstacle_distance(points)

    def sheet_distance(self, points):
        return self.meshes.sheet_distance(points)

    def avoid_distance(self, points):
        return self.meshes.avoid_distance(points)

    def summary(self):
        return dict(
            mode=self.mode,
            part_category=self.part_category,
            params=self.params.to_dict(),
            n_start=1 if self.endpoints.start else 0,
            n_end=self.endpoints.n_endpoints,
            n_control=self.endpoints.n_controls,
            stl=self.meshes.summary(),
            n_routing_surface=self.surfaces.n_routing,
            n_avoidance_surface=self.surfaces.n_avoidance,
        )


def load_environment(case_dir):
    return Environment(case_dir)
