# 60mm卡扣筛选历史快照（穿行修复之前）

本分支对应“卡扣像终点/末端折返”的核查时点，目的为保留当时原样代码，不是修复版本。

## 来源

基线提交：b5c87f7e1dacf3831d12dff61de6e4b45bd23a80。
route.py与7个harness_plan模块逐字节匹配55服务器spacing60实验payload_manifest中的SHA-256。指纹见spacing60_source_manifest.json。
普通before_transit备份的route.py与服务器入口不一致，因此选用核对一致的route_spacing60.py恢复为route.py；其余7模块匹配。
当前改进版另存于cmx/routing-correctness分支bfb75e1082c2a33400e3686ba161ee132f1948da。

## 行为与已知问题

按规划网络沿线距离执行60mm候选卡扣排除。CPU与GPU的几何路径不同，因此排除数量不同；不合理折返也可能放大沿线距离。
GPU卡扣单入口网格接入和拓扑绕行允许卡扣成为悬挂折返；CPU也有个别同类问题。位置覆盖统计不等于有效穿行；此快照没有后续的独立分叉、卡扣两侧方向门段与最终l5反馈重构，不包含validation_layer.py。

当时60mm结果核对：

|案例|通道|计为使用的客户卡扣|末端折返卡扣|
|---|---|---:|---:|
|B07|CPU|10|0|
|B07|GPU|11|11|
|C07|CPU|11|1（REF-009）|
|C07|GPU|14|13|

这些是历史结果，不是本次重新运行结果。案例为当时B07及C07三终点候选卡扣测试集，不是后续补齐的C07二十二终点版本。

## 使用

安装requirements.txt基础依赖；GPU另需相应PyTorch环境。

```sh
python route.py --case <original_case_directory> --device cpu --voxel 30 --out <result.json>
python route.py --case <original_case_directory> --device gpu --voxel 30 --out <result.json>
```

CUDA不可用时原型可能退回CPU，device标签不能代替真实后端核验。客户输入、STL和结果不包含在本分支。

本次只恢复并核对8个算法源码文件，不加入当前改进版测试、查看器、服务器运行脚本或历史备份，不重新测试或修改缺陷。

可复用经验：历史版本必须与当次运行指纹逐文件核对，不能仅按备份目录名称认定版本一致。
