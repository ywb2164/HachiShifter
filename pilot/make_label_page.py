"""results_v2.json → 点选标注页 label.html。"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / "results_v2.json").read_text(encoding="utf-8"))
slim = [{k: r.get(k) for k in ("id", "folder", "file", "initial", "initial_kind", "start", "end", "p", "duration")} for r in rows]

html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>P 点标注</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#16151b;color:#e8e6ef}
 header{position:sticky;top:0;background:#201e28;padding:10px 16px;border-bottom:1px solid #333;z-index:5}
 header h1{font-size:16px;margin:0 0 4px} header p{margin:2px 0;font-size:13px;color:#b9b5c8}
 #stats{font-size:13px;color:#7dd3a0}
 .card{background:#201e28;border:1px solid #333;border-radius:8px;margin:10px 12px;padding:8px}
 .card.done{border-color:#3a7}
 .spec{position:relative;width:100%;cursor:crosshair;user-select:none}
 .spec img{width:100%;display:block;border-radius:4px}
 .line{position:absolute;top:0;bottom:0;width:2px;pointer-events:none}
 .auto{background:#ffd400}.mine{background:#3ef08a;width:3px}.hover{background:rgba(255,255,255,.55);width:1px}
 .shade{position:absolute;top:0;bottom:0;background:rgba(255,212,0,.13);pointer-events:none}
 .meta{font-size:13px;margin:4px 0}
 button{font-family:inherit;font-size:13px;margin:2px;padding:4px 9px;border-radius:5px;border:1px solid #555;background:#2c2a36;color:#e8e6ef;cursor:pointer}
 .pl{background:#263240} button.on{background:#7f69ca}
 .off{color:#ffd400;font-size:13px;margin-left:6px}
</style></head><body>
<header><h1>P 点标注：在频谱图上点出你认为正确的辅音/元音交界</h1>
<p><b style="color:#ffd400">黄线</b> = 新算法的 P（黄色阴影是辅音段）；<b style="color:#3ef08a">绿线</b> = 你点的位置。点击频谱图任意处即可设置绿线，可反复点。</p>
<p>试听：<b>▶从绿线</b> 应当直接是元音；<b>▶到绿线</b> 应当只有辅音。黄线已经对了就点“黄线就对”。素材本身缺辅音、有杂音等点“素材问题”。<b>每次操作都会自动保存</b>，标 20～30 个就够用。</p>
<div id="stats"></div></header>
<div id="list"></div>
<script>
const ROWS = __ROWS__;
let labels = {};
const ctx = new AudioContext(); const bufs = {}; let cur = null;
async function buf(id){ if(!bufs[id]){ const r = await fetch("audio/"+id+".wav"); bufs[id] = await ctx.decodeAudioData(await r.arrayBuffer()); } return bufs[id]; }
async function play(id, a, b){ if(cur){try{cur.stop()}catch(e){}} await ctx.resume(); const s = ctx.createBufferSource(); s.buffer = await buf(id); s.connect(ctx.destination);
  a = Math.max(0, a ?? 0); b == null ? s.start(0, a) : s.start(0, a, Math.max(0.01, b - a)); cur = s; }
async function save(){ await fetch("/labels?set=p_single", {method:"POST", body: JSON.stringify(labels)}); stats(); }
function stats(){
  const done = ROWS.filter(r => labels[r.id]); const mats = done.filter(r => labels[r.id].bad).length;
  const offs = done.filter(r => !labels[r.id].bad && r.p != null).map(r => (labels[r.id].p - r.p) * 1000);
  const within = offs.filter(x => Math.abs(x) <= 15).length;
  const med = offs.length ? offs.slice().sort((a,b)=>a-b)[Math.floor(offs.length/2)].toFixed(0) : "-";
  document.getElementById("stats").textContent = `已标 ${done.length}/${ROWS.length}（素材问题 ${mats}）　|　黄线与你的差：中位 ${med} ms，±15 ms 以内 ${within}/${offs.length}`;
}
const mk = (t, f, cls) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; if(cls) b.className = cls; return b; };
function card(r){
  const c = document.createElement("div"); c.className = "card"; c.id = "c-"+r.id;
  const pct = t => (100 * t / r.duration) + "%";
  c.innerHTML = `<div class="meta">${r.id} · ${r.folder} / <b>${r.file}</b> · 声母 ${r.initial || "零"}<span class="off"></span></div>
    <div class="spec"><img src="img2/${r.id}.png"><div class="shade"></div><div class="line auto"></div><div class="line mine" style="display:none"></div><div class="line hover" style="display:none"></div></div>
    <div class="btns"></div>`;
  const spec = c.querySelector(".spec"), mine = c.querySelector(".mine"), hover = c.querySelector(".hover"), off = c.querySelector(".off");
  if(r.p != null){ c.querySelector(".auto").style.left = pct(r.p); const sh = c.querySelector(".shade"); sh.style.left = pct(r.start); sh.style.width = (100*(r.p - r.start)/r.duration) + "%"; }
  else c.querySelector(".auto").style.display = "none";
  const refresh = () => { const L = labels[r.id]; c.classList.toggle("done", !!L);
    if(L && !L.bad){ mine.style.display = "block"; mine.style.left = pct(L.p); off.textContent = r.p != null ? `你的点比黄线${L.p >= r.p ? "晚" : "早"} ${Math.abs((L.p - r.p)*1000).toFixed(0)} ms` : ""; }
    else { mine.style.display = "none"; off.textContent = L && L.bad ? "已标：素材问题" : ""; } };
  spec.onmousemove = e => { const rc = spec.getBoundingClientRect(); hover.style.display = "block"; hover.style.left = (e.clientX - rc.left) + "px"; };
  spec.onmouseleave = () => hover.style.display = "none";
  spec.onclick = e => { const rc = spec.getBoundingClientRect(); const t = (e.clientX - rc.left) / rc.width * r.duration;
    labels[r.id] = {p: +t.toFixed(4)}; refresh(); save(); play(r.id, t); };
  const b = c.querySelector(".btns");
  b.append(mk("▶ 整个", () => play(r.id), "pl"));
  b.append(mk("▶ 从黄线", () => play(r.id, r.p), "pl"));
  b.append(mk("▶ 从绿线", () => labels[r.id] && !labels[r.id].bad && play(r.id, labels[r.id].p), "pl"));
  b.append(mk("▶ 到绿线", () => labels[r.id] && !labels[r.id].bad && play(r.id, r.start, labels[r.id].p), "pl"));
  b.append(mk("黄线就对", () => { if(r.p != null){ labels[r.id] = {p: r.p, same: true}; refresh(); save(); } }));
  b.append(mk("素材问题", () => { labels[r.id] = {bad: true}; refresh(); save(); }));
  b.append(mk("清除", () => { delete labels[r.id]; refresh(); save(); }));
  refresh(); return c;
}
(async () => { labels = await (await fetch("/labels?set=p_single")).json(); const list = document.getElementById("list"); ROWS.forEach(r => list.append(card(r))); stats(); })();
</script></body></html>"""
(OUT / "label.html").write_text(html.replace("__ROWS__", json.dumps(slim, ensure_ascii=False)), encoding="utf-8")
print("written", OUT / "label.html")
