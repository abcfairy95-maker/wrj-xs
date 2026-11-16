"""生成结果 3D 可视化 HTML（障碍 + 贴面 + 线束路径 + 卡扣 + 端点）。

用法：
  python view_result.py --case "D:/下载/datas/构造/C07_candidate_clips_3ends_20261112_15clips" \
        --result output/C07_result_new.json --out output/C07_view.html
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from route import load_case


def mesh_subsample(files, max_faces=20000):
    """按面抽样并只保留被引用顶点，控制 HTML 体积。"""
    verts, faces = [], []
    budget = max(1, max_faces // max(1, len(files)))
    off = 0
    for f in files:
        try:
            m = trimesh.load_mesh(f, process=False)
            V = np.asarray(m.vertices)
            F = np.asarray(m.faces)
            step = max(1, len(F) // budget)
            Fs = F[::step]
            used = np.unique(Fs.ravel())
            remap = {int(o): i for i, o in enumerate(used)}
            verts.extend([[round(float(x), 2) for x in p] for p in V[used]])
            faces.extend([[int(off + remap[int(a)]), int(off + remap[int(b)]),
                           int(off + remap[int(c)])] for a, b, c in Fs])
            off += len(used)
        except Exception:
            pass
    return verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--input-file', default=None)
    ap.add_argument('--result', required=True)
    ap.add_argument('--out', default='output/view.html')
    args = ap.parse_args()

    case = load_case(args.case, input_file=args.input_file)
    res = json.load(open(args.result, encoding='utf-8'))

    paths = []
    colors = ['#ff8c00', '#00c2ff', '#ffdd44', '#ff6688', '#66ff99']
    for i, (hid, pts) in enumerate(res['harness_paths'].items()):
        eid = next((h['endpoint_id'] for h in res['topology']['harnesses'] if h['id'] == hid), hid)
        paths.append(dict(id=hid, end=eid, color=colors[i % len(colors)], pts=pts))

    clips = [dict(id=c['id'], pos=[round(float(x), 3) for x in c['pos']],
                  dir=[round(float(x), 3) for x in c['dir']]) for c in case['clips']]
    start = dict(id=case['start']['id'], pos=[round(float(x), 3) for x in case['start']['pos']],
                 dir=[round(float(x), 3) for x in case['start']['dir']])
    ends = [dict(id=e['id'], pos=[round(float(x), 3) for x in e['pos']],
                 dir=[round(float(x), 3) for x in e['dir']]) for e in case['ends']]

    o_verts, o_faces = mesh_subsample(case['env'])
    s_verts, s_faces = [], []

    data = dict(paths=paths, clips=clips, start=start, ends=ends,
                obstacle=dict(vertices=o_verts, faces=o_faces),
                sheet=dict(vertices=s_verts, faces=s_faces),
                stats=res.get('statistics', {}))

    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>布线结果</title>
<style>body{margin:0;overflow:hidden;font-family:monospace}#info{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.8);color:#0f0;padding:8px;font-size:13px;border-radius:4px;max-width:420px}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script></head><body>
<div id="info"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA=__DATA__;
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0e0e1a);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,1,30000);
camera.position.set(1600,900,2200);
const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setSize(innerWidth,innerHeight);
document.body.appendChild(renderer.domElement);
new OrbitControls(camera,renderer.domElement);
scene.add(new THREE.GridHelper(6000,60,0x2a2a3a,0x181824));
scene.add(new THREE.HemisphereLight(0xffffff,0x334,1.2));
scene.add(new THREE.DirectionalLight(0xffffff,0.8));
function addMesh(verts,faces,color,opacity){
  if(!verts.length) return;
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(verts.flat(),3));
  g.setIndex(faces.flat()); g.computeVertexNormals();
  scene.add(new THREE.Mesh(g,new THREE.MeshStandardMaterial({color,transparent:true,opacity,roughness:.7,metalness:.05,side:THREE.DoubleSide})));
}
addMesh(DATA.obstacle.vertices,DATA.obstacle.faces,0x9fb2c8,0.35);
addMesh(DATA.sheet.vertices,DATA.sheet.faces,0x3388ff,0.25);
// 线束路径（管道）
for(const p of DATA.paths){
  const pts=p.pts.map(v=>new THREE.Vector3(...v));
  const curve=new THREE.CatmullRomCurve3(pts,false,'catmullrom',0.1);
  const geo=new THREE.TubeGeometry(curve,Math.min(400,pts.length*2),2.5,8,false);
  scene.add(new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:p.color,roughness:.4,metalness:.3})));
  // 端点标签球
  const tail=pts[pts.length-1];
  const b=new THREE.Mesh(new THREE.SphereGeometry(9,12,12),new THREE.MeshBasicMaterial({color:p.color}));
  b.position.copy(tail); scene.add(b);
}
// 起点（绿）与终点（蓝）
const st=new THREE.Mesh(new THREE.SphereGeometry(11,16,16),new THREE.MeshBasicMaterial({color:0x00ff66}));
st.position.set(...DATA.start.pos); scene.add(st);
scene.add(new THREE.ArrowHelper(new THREE.Vector3(...DATA.start.dir),st.position,80,0x00ff66,12,8));
for(const e of DATA.ends){
  const b=new THREE.Mesh(new THREE.SphereGeometry(10,16,16),new THREE.MeshBasicMaterial({color:0xff4444}));
  b.position.set(...e.pos); scene.add(b);
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(...e.dir),b.position,70,0xff4444,10,7));
}
// 卡扣（黄球 + 方向箭头）
for(const c of DATA.clips){
  const b=new THREE.Mesh(new THREE.SphereGeometry(6,12,12),new THREE.MeshBasicMaterial({color:0xffff00}));
  b.position.set(...c.pos); scene.add(b);
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(...c.dir),b.position,55,0xffff00,8,6));
}
document.getElementById('info').textContent='线束='+DATA.paths.length+' 卡扣='+DATA.clips.length+
  ' 终点='+DATA.ends.length+' | 总长='+(DATA.stats.total_length_mm||0).toFixed(0)+'mm';
function loop(){requestAnimationFrame(loop);renderer.render(scene,camera);} loop();
</script></body></html>"""
    html = html.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding='utf-8')
    print('生成', args.out, '%.1f KB' % (Path(args.out).stat().st_size / 1024))


if __name__ == '__main__':
    main()
