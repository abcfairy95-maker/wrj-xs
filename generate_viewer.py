"""生成案例环境的 3D 可视化 HTML（障碍 + 不规则贴面实体 sheet_part + 端点 + 贴面区点）。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, '.')
from environment import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='D:/下载/datas/one click/C01 电池包STAN')
    parser.add_argument('--output', type=str, default='output/environment_viewer.html')
    args = parser.parse_args()

    case = Path(args.case)
    if not case.exists():
        # 按前缀在 data 下匹配真实路径（中文名可能在终端显示乱码）
        for m in ['one click', 'CPR']:
            d = Path('D:/下载/datas') / m
            if not d.is_dir():
                continue
            for c in d.iterdir():
                if c.is_dir() and c.name.startswith(case.name):
                    case = c
                    break
    env = Environment(case)

    points = []
    for n in [env.endpoints.start] + env.endpoints.ends + env.endpoints.controls:
        if n is None:
            continue
        points.append({'id': n.id, 'pos': [round(float(x), 3) for x in n.pos],
                       'dir': [round(float(x), 4) for x in n.dir], 'type': n.kind})

    # 贴面区点（routing_surfaces）
    surf_points = []
    for s in env.surfaces.routing_surfaces:
        surf_points.append({'id': s.part_id, 'pos': [round(float(x), 3) for x in s.point],
                            'dir': [round(float(x), 4) for x in s.dir], 'direction': s.direction})

    def mesh_arrays(files):
        verts, faces = [], []
        for f in files:
            try:
                m = trimesh.load_mesh(f, process=False)
                off = len(verts)
                verts.extend([[round(float(x), 2) for x in v] for v in m.vertices])
                faces.extend([[int(a + off), int(b + off), int(c + off)] for a, b, c in m.faces])
            except Exception as e:
                print('加载失败', f.name, e)
        return verts, faces

    o_verts, o_faces = mesh_arrays(env.meshes.files['env'])
    s_verts, s_faces = mesh_arrays(env.meshes.files['sheet'])

    data = {'points': points, 'surf_points': surf_points,
            'obstacle': {'vertices': o_verts, 'faces': o_faces},
            'sheet': {'vertices': s_verts, 'faces': s_faces},
            'stats': {'n_obstacle': env.meshes.summary()['env'], 'n_sheet': env.meshes.summary()['sheet'],
                      'n_end': env.endpoints.n_endpoints, 'n_surf': env.surfaces.n_routing,
                      'min_bend_radius': env.params.min_bend_radius,
                      'clearance': env.params.clearance_line, 'case': Path(args.case).name}}

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>环境可视化</title>
<style>body{margin:0;overflow:hidden;font-family:monospace}#info{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);color:#0f0;padding:8px;font-size:13px;border-radius:4px}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="info"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA = __DATA__;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0e1a);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 1, 30000);
camera.position.set(2500, 1200, 2500);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.GridHelper(6000, 60, 0x2a2a3a, 0x181824));
scene.add(new THREE.HemisphereLight(0xffffff,0x334,1.1));
scene.add(new THREE.DirectionalLight(0xffffff,0.7));
function addMesh(verts, faces, color, opacity){
  if(!verts.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts.flat(), 3));
  g.setIndex(faces.flat());
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color:color,transparent:true,opacity:opacity,roughness:0.6,metalness:0.05,side:THREE.DoubleSide}));
  scene.add(m);
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), new THREE.LineBasicMaterial({color:color})));
}
// 障碍（灰蓝实心）
addMesh(DATA.obstacle.vertices, DATA.obstacle.faces, 0x9fb2c8, 0.9);
// 贴面实体 sheet_part（蓝色不规则面）
addMesh(DATA.sheet.vertices, DATA.sheet.faces, 0x3388ff, 0.5);
// 贴面区点（蓝）+ 方向
for(const s of DATA.surf_points){
  const ball = new THREE.Mesh(new THREE.SphereGeometry(10,12,12), new THREE.MeshBasicMaterial({color:0x44ccff}));
  ball.position.set(...s.pos); scene.add(ball);
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(...s.dir), new THREE.Vector3(...s.pos), 80, 0x44ccff, 10, 7));
}
// 端点
for(const p of DATA.points){
  const c = p.type==='start'?0x00ff66:0xff4444;
  const ball = new THREE.Mesh(new THREE.SphereGeometry(9,16,16), new THREE.MeshBasicMaterial({color:c}));
  ball.position.set(...p.pos); scene.add(ball);
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(...p.dir), new THREE.Vector3(...p.pos), 70, c, 9, 7));
}
document.getElementById('info').textContent =
  DATA.stats.case+' | 障碍='+DATA.stats.n_obstacle+' 贴面='+DATA.stats.n_sheet+' 终点='+DATA.stats.n_end+
  ' 贴面区点='+DATA.stats.n_surf+' | 转弯半径='+DATA.stats.min_bend_radius+'mm 净空='+DATA.stats.clearance+'mm';
function loop(){requestAnimationFrame(loop);renderer.render(scene,camera);}
loop();
</script></body></html>"""

    html = html.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    out = Path(args.output)
    out.write_text(html, encoding='utf-8')
    print('生成', out)
    print('障碍面=%d 贴面面=%d 端点=%d 贴面点=%d' % (len(o_faces), len(s_faces), len(points), len(surf_points)))
    print('HTML %.1f KB' % (out.stat().st_size / 1024))


if __name__ == '__main__':
    main()
