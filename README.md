# Harness Routing（汽车线束三维布线）

## 当前进度：固定卡扣辅助布线研究原型

已增加CPU/GPU布线入口及固定卡扣穿行、独立分叉和最终几何校验。**仍存在转弯半径违规，尚未生产验收。**

使用方法、实际算法、案例结果及验证限制见 [交付说明](docs/routing_prototype_delivery.md)。推荐使用 `route.py` 命令行入口，客户数据与结果不入库。

## 原有地图构建模块

以下为原有 `environment` 数据加载模块说明；新增布线算法位于 `harness_plan`。

## 目录结构

```
harness_routing/
├── environment/              # 地图构建层（总文件夹）
│   ├── __init__.py           # Environment 统一入口
│   ├── params.py             # 参数：线径/半径/净空/间距 + 范围校验
│   ├── points.py             # 端点 + 途径点（位置/方向/顺序）
│   ├── meshes.py             # 三类 STL：障碍(env_)/贴面(sheet_part)/避让(avoid_part)
│   └── surfaces.py           # 贴面区(routing_surfaces) + 局部避让面(avoidance_surfaces)
├── docs/
│   ├── parameters.md         # 参数定义（符号/范围/默认值）
│   └── input_schema.md       # 输入格式（input.json 字段）
├── cases/                    # 测试案例数据
├── output/                   # 输出
└── requirements.txt          # numpy / trimesh
```

## 使用

```python
from environment import Environment

env = Environment('D:/下载/datas/one click/C01 电池包STAN')

env.params             # 参数（D_t/d_b/r/d/l1/l2/x/l3~l6 + 派生量 + 校验 warnings）
env.endpoints          # 端点（start/ends/controls，connections() 返回连接）
env.meshes             # 三类 STL（obstacle/sheet/avoid + 距离查询）
env.surfaces           # 贴面区 + 避让面

# 距离查询
env.obstacle_distance(points)   # 障碍距离（硬避障）
env.sheet_distance(points)      # 贴面距离（软约束）
env.avoid_distance(points)      # 避让距离（软约束）
```

## 支持的数据

| 数据 | 类型 | 状态 |
|---|---|---|
| 障碍 | `env_` STL | ✅ 21 案例全有 |
| 贴面 | `sheet_part` STL + `routing_surfaces` | ✅ 11 案例有 |
| 避让 | `avoid_part` STL + `avoidance_surfaces` | ⚠️ 数据里 0 个 |
| 端点 | `endpoints`（pos + dir）| ✅ |
| 途径点 | `control_points`（pos + dir + order）| ✅ |
| 参数 | `preferences`（线径/半径/净空/间距）| ✅ 含范围校验 |
