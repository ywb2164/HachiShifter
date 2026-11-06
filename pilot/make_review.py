"""把 results.json 生成为试听打勾网页 review.html（结果内嵌，不需要额外请求）。"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>单字 P 点试听审核</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#16151b;color:#e8e6ef}
 header{position:sticky;top:0;background:#201e28;padding:10px 16px;border-bottom:1px solid #333;z-index:2}
 header h1{font-size:16px;margin:0 0 4px} header p{margin:2px 0;font-size:13px;color:#b9b5c8}
 #stats{font-size:13px;color:#ffd400}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:12px;padding:12px}
 .card{background:#201e28;border:1px solid #333;border-radius:8px;padding:8px}
 .card.done{border-color:#3a7}
 .card img{width:100%;border-radius:4px;display:block}
 .meta{font-size:13px;margin:4px 0}.doubt{color:#ff8a65}.sure{color:#7dd3a0}
 button{font-family:inherit;font-size:13px;margin:2px;padding:4px 9px;border-radius:5px;border:1px solid #555;background:#2c2a36;color:#e8e6ef;cursor:pointer}
 button.on{background:#7f69ca;border-color:#a593ef}
 .play button{background:#263240}
</style></head><body>
<header><h1>单字素材 P 点（辅音/元音交界）试听审核</h1>
<p>黄色竖线 = 自动找到的 P；黄色阴影 = 辅音段。<b>“只听辅音”</b>应当听不到元音，<b>“从 P 开始”</b>应当直接是元音、听不到辅音。</p>
<p>判断：<b>准</b>（差别听不出来，能直接用）· <b>偏早</b>（从 P 开始还能听到辅音）· <b>偏晚</b>（只听辅音时混进了元音）· <b>素材问题</b>（素材本身缺音/杂音/读错，与 P 无关）</p>
<div id="stats"></div></header>
<div class="grid" id="grid"></div>
<script>
const ROWS = __ROWS__;
const KEY = "pilot-p-review-v1";
const verdicts = JSON.parse(localStorage.getItem(KEY) || "{}");
const ctx = new (window.AudioContext || window.webkitAudioContext)();
const buffers = {};
let current = null;
async function buf(id){ if(!buffers[id]){ const r = await fetch("audio/"+id+".wav"); buffers[id] = await ctx.decodeAudioData(await r.arrayBuffer()); } return buffers[id]; }
async function play(id, from, to){
  if(current){ try{current.stop()}catch(e){} }
  await ctx.resume(); const b = await buf(id); const s = ctx.createBufferSource(); s.buffer = b; s.connect(ctx.destination);
  const a = Math.max(0, from ?? 0), d = to==null ? undefined : Math.max(0.01, to - a);
  d===undefined ? s.start(0, a) : s.start(0, a, d); current = s;
}
const LABELS = {ok:"准", early:"偏早", late:"偏晚", bad:"素材问题"};
function stats(){
  const done = ROWS.filter(r => verdicts[r.id]);
  const by = (filt) => { const xs = done.filter(filt); const ok = xs.filter(r => verdicts[r.id]==="ok").length; const bad = xs.filter(r => verdicts[r.id]==="bad").length; return {n:xs.length, ok, bad}; };
  const s = by(r => r.verdict==="自信"), d = by(r => r.verdict==="存疑");
  const pct = (a,b) => b ? (100*a/b).toFixed(0)+"%" : "-";
  document.getElementById("stats").textContent =
    `已审 ${done.length}/${ROWS.length}　|　自信组：${s.n} 个，P 准 ${s.ok}（${pct(s.ok,s.n-s.bad)}，不计素材问题）　|　存疑组：${d.n} 个，P 准 ${d.ok}，素材问题 ${d.bad}`;
}
function render(){
  const g = document.getElementById("grid");
  for(const r of ROWS){
    const c = document.createElement("div"); c.className = "card" + (verdicts[r.id] ? " done" : ""); c.id = "c-"+r.id;
    const reason = r.reasons && r.reasons.length ? "（" + r.reasons.join("；") + "）" : "";
    c.innerHTML = `<img src="img/${r.id}.png" loading="lazy">
      <div class="meta">${r.id} · ${r.folder} / <b>${r.file}</b> · 声母 ${r.initial||"零"} · 辅音 ${r.consonant_ms ?? "-"} ms ·
      <span class="${r.verdict==='自信'?'sure':'doubt'}">${r.verdict}${reason}</span></div>
      <div class="play"></div><div class="judge"></div>`;
    const p = c.querySelector(".play");
    const mk = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; return b; };
    p.append(mk("▶ 整个", () => play(r.id)));
    if(r.p != null){ p.append(mk("▶ 只听辅音", () => play(r.id, r.start, r.p)), mk("▶ 从 P 开始", () => play(r.id, r.p))); }
    const j = c.querySelector(".judge");
    for(const [k, t] of Object.entries(LABELS)){
      const b = mk(t, () => { verdicts[r.id] = k; localStorage.setItem(KEY, JSON.stringify(verdicts));
        j.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); c.classList.add("done"); stats(); });
      if(verdicts[r.id] === k) b.classList.add("on"); j.append(b);
    }
    g.append(c);
  }
  stats();
}
render();
window.exportVerdicts = () => JSON.stringify(verdicts);
</script></body></html>"""
(OUT / "review.html").write_text(html.replace("__ROWS__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
print("written", OUT / "review.html")
