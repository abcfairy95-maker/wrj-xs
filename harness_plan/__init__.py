"""汽车线束布线 —— 四层架构（CPU / GPU 双通道，共享数据/拓扑/结果层）。

层结构：
  数据层  data_layer     load_case / params / Collision        （共享）
  地图层  map_layer      Roadmap 体素路图（含 device 分发）
  寻路层  pathfind_layer CPU Dijkstra/A* / GPU 波前传播
  拓扑层  topology_layer steiner：MST + 终点挂叶子 + 卡扣强穿   （共享）
  平滑层  smooth_layer   CPU Hermite / GPU Adam 可微分
  结果层  result_layer   build_topology → 完整 result.json      （共享）

入口在根目录 route.py：route(case, device='cpu'|'gpu')。
"""
from . import data_layer, map_layer, pathfind_layer, topology_layer, smooth_layer, result_layer

__all__ = ['data_layer', 'map_layer', 'pathfind_layer',
           'topology_layer', 'smooth_layer', 'result_layer']
