"""sentences.json → 整句切字试听审核页 sentences.html。"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / "sentences.json").read_text(encoding="utf-8"))

html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>整句切字试听审核</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#16151b;color:#e8e6ef}
 header{position:sticky;top:0;background:#201e28;padding:10px 16px;border-bottom:1px solid #333;z-index:2}
 header h1{font-size:16px;margin:0 0 4px} header p{margin:2px 0;font-size:13px;color:#b9b5c8}
 #stats{font-size:13px;color:#ffd400}
 .sent{background:#201e28;border:1px solid #333;border-radius:8px;margin:12px;padding:8px}
 .sent img{max-width:100%;border-radius:4px;display:block}
 .meta{font-size:13px;margin:4px 0}.bad{color:#ff8a65}.good{color:#7dd3a0}
 .chars{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
 .ch{border:1px solid #444;border-radius:6px;padding:4px 6px;font-size:13px;min-width:150px}
 .ch.done{border-color:#3a7}.ch .t{font-size:18px;margin-right:4px}
 button{font-family:inherit;font-size:12px;margin:1px;padding:3px 7px;border-radius:5px;border:1px solid #555;background:#2c2a36;color:#e8e6ef;cursor:pointer}
 button.on{background:#7f69ca;border-color:#a593ef} .pl{background:#263240}
</style></head><body>
<header><h1>整句素材自动切字审核</h1>
<p>每个字：<b>▶字</b> 播放这个字的范围（应当正好是这个字，不缺头不带尾）；<b>▶辅音</b> 应当只有辅音；<b>▶P后</b> 应当直接是元音。</p>
<p>判断：<b>准</b>（能直接当素材用）· <b>边界偏</b>（是这个字但范围或 P 不对）· <b>字不对</b>（切出来不是这个字）。不必全审，随便审几十个字即可。</p>
<div id="stats"></div></header>
<div id="list"></div>
<script>
const ROWS = __ROWS__;
const KEY = "pilot-sentence-review-v1";
let v = {};
const ctx = new AudioContext(); const bufs = {}; let cur = null;
async function buf(id){ if(!bufs[id]){ const r = await fetch("sent_audio/"+id+"_play.wav"); bufs[id] = await ctx.decodeAudioData(await r.arrayBuffer()); } return bufs[id]; }
async function play(id, a, b){ if(cur){try{cur.stop()}catch(e){}} await ctx.resume(); const s = ctx.createBufferSource(); s.buffer = await buf(id); s.connect(ctx.destination);
  a = Math.max(0, a ?? 0); b == null ? s.start(0, a) : s.start(0, a, Math.max(0.01, b - a)); cur = s; }
const L = {ok:"准", edge:"边界偏", wrong:"字不对"};
function stats(){
  const all = ROWS.flatMap(r => r.chars.map((c,i) => ({...c, key: r.id+":"+i})));
  const done = all.filter(c => v[c.key]);
  const grp = f => { const xs = done.filter(f); return [xs.length, xs.filter(c => v[c.key]==="ok").length]; };
  const [sn, so] = grp(c => c.verdict==="自信"), [dn, dok] = grp(c => c.verdict==="存疑");
  const pct = (a,b) => b ? (100*a/b).toFixed(0)+"%" : "-";
  document.getElementById("stats").textContent = `已审 ${done.length} 字 / 共 ${all.length} 字　|　自信组 ${sn} 字，准 ${so}（${pct(so,sn)}）　|　存疑组 ${dn} 字，准 ${dok}（${pct(dok,dn)}）`;
}
const mk = (t, f, cls) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; if(cls) b.className = cls; return b; };
(async () => { v = await (await fetch("/labels?set=sentences_v2")).json();
for(const r of ROWS){
  const d = document.createElement("div"); d.className = "sent";
  d.innerHTML = `<div class="meta">${r.id} · ${r.folder} / <b>${r.file}</b><br>文件名（仅参考）「${r.name_text||""}」 → 录音实际「${r.text}」${r.dropped ? " · 文件名中录音里没有：「"+r.dropped+"」" : ""} ·
    <span class="${r.status==='ok'?'good':'bad'}">${r.status}</span>　<span style="color:#7dd3a0">绿字</span>=两边一致　<span style="color:#ff8a65">橙字</span>=只有录音里有</div>`;
  if(r.chars.length){ const img = document.createElement("img"); img.src = "sent_img/"+r.id+".png"; img.loading = "lazy"; d.append(img); }
  d.append(mk("▶ 整句", () => play(r.id), "pl"));
  const box = document.createElement("div"); box.className = "chars";
  r.chars.forEach((c, i) => {
    const key = r.id+":"+i; const e = document.createElement("div"); e.className = "ch" + (v[key] ? " done" : "");
    e.innerHTML = `<span class="t" style="color:${c.source==='both'?'#7dd3a0':'#ff8a65'}">${c.char}</span>${c.pinyin} <span class="${c.verdict==='自信'?'good':'bad'}" title="${c.reasons.join('；')}">${c.verdict}</span><br>`;
    e.append(mk("▶字", () => play(r.id, c.start, c.end), "pl"));
    if(c.p != null){ e.append(mk("▶辅音", () => play(r.id, c.start, c.p), "pl"), mk("▶P后", () => play(r.id, c.p, c.end), "pl")); }
    e.append(document.createElement("br"));
    for(const [k, t] of Object.entries(L)){ const b = mk(t, () => { v[key] = k; fetch("/labels?set=sentences_v2", {method:"POST", body: JSON.stringify(v)});
        e.querySelectorAll("button:not(.pl)").forEach(x => x.classList.remove("on")); b.classList.add("on"); e.classList.add("done"); stats(); });
      if(v[key] === k) b.classList.add("on"); e.append(b); }
    box.append(e);
  });
  d.append(box); document.getElementById("list").append(d);
}
stats(); })();
window.exportVerdicts = () => JSON.stringify(v);
</script></body></html>"""
(OUT / "sentences.html").write_text(html.replace("__ROWS__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
print("written", OUT / "sentences.html")
