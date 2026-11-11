# 输入格式（input.json Schema）

## 顶层结构

```json
{
  "avoidance_surfaces": [],
  "control_points": [...],      // 仅 control_point 模式
  "endpoints": { "starts": [...], "ends": [...] },
  "mode": "one_click" | "control_point",
  "part_category": "body" | "front_compartment" | ...,
  "preferences": { ... },
  "routing_direction": { "routing_surfaces": [...] },
  "schema_version": "2.0"
}
```

## endpoints.starts / ends（端点）

```json
{
  "dir": [x, y, z],          // 出线方向（单位向量）
  "hasElectricalAttribute": true,
  "id": "START-001" | "END-001",
  "pos": [x, y, z],          // 位置（mm）
  "sourcePartId": "..."
}
```

## control_points（途径点，仅 control_point 模式）

```json
{
  "dir": [x, y, z],          // 遍历切向
  "is_existing_clip": false,
  "order": 1,                // 连接顺序
  "point_id": "CP-001",
  "pos": [x, y, z]
}
```

## preferences（参数）

```json
{
  "bend_radius_factor": 2,          // r，转弯半径倍数
  "clearance_mm": 6,                // d，避障距离
  "max_harness_diameter_mm": 12,    // D_t，主干线径
  "min_harness_diameter_mm": 6,     // d_b，分支线径
  "terminal_branch_min_length_mm": 60,   // l1
  "split_point_min_spacing_mm": 75,      // l2
  "start_to_first_fork_min_distance_mm": 40,
  "max_fork_degree": 5,             // x
  "clip_max_spacing_mm": 150,       // l4
  "clip_min_spacing_mm": 100,       // l3
  "clip_min_distance_to_split_mm": 40,   // l5
  "clip_template_id": "PN-CLIP-001",
  "generate_clips": true
}
```

## routing_direction.routing_surfaces（贴面区）

```json
{
  "dir": [x, y, z],          // 该贴面区域线束走向方向
  "direction": "X-" | "Y-" | "Z+" | "Y+" | "CUSTOM",
  "part_id": "001",
  "point_on_surface": [x, y, z]
}
```

## 障碍（STL）

- 位置：`case/input/` 下，`env_` 前缀的 `.stl` 文件
- 用途：避障（线束外表面到障碍 ≥ d）

## 两种模式的连接语义

| 模式 | 连接 |
|---|---|
| one_click | START → 每个 END（星型，N 条）|
| control_point | START → CP-1 → CP-2 → ... → END（链式，按 order 排序）|

方向约定：终点 `dir` 为「朝外」方向，到达方向 = `-dir`；中间控制点 `dir` 为「遍历切向」。
