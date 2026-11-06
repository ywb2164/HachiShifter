"""盲标页：不显示算法结果，40 个新单字 + 10 个上次标过的（打乱，用于测人自身一致性）。

输出 audio_blind/、img_blind/、blind.json、label_blind.html；标注保存到 labels_p_blind.json。
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import json
import random
from pathlib import Path

import librosa
import soundfile as sf

from detect_v2 import SR, plot

OUT = Path(__file__).resolve().parent
LIB = cfg.LIBRARY
rnd = random.Random(2026)

old = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
used = {(r["folder"], r["file"]) for r in old}
labels = json.loads((OUT / "labels_p_single.json").read_text(encoding="utf-8"))

items = []
for folder in ("棍の音/棍音源", "张顺飞素材/张顺飞活字印刷"):
    files = sorted(p for p in (LIB / folder).iterdir() if p.suffix.lower() == ".wav" and (folder, p.name) not in used)
    for f in rnd.sample(files, 20):
        items.append({"folder": folder, "file": f.name, "src": str(f), "repeat_of": None})
repeat_ids = rnd.sample(sorted(k for k, v in labels.items() if not v.get("bad")), 10)
by_id = {r["id"]: r for r in old}
for k in repeat_ids:
    r = by_id[k]
    items.append({"folder": r["folder"], "file": r["file"], "src": str(LIB / r["folder"] / r["file"]), "repeat_of": k})
rnd.shuffle(items)

(OUT / "audio_blind").mkdir(exist_ok=True)
(OUT / "img_blind").mkdir(exist_ok=True)
for i, it in enumerate(items):
    bid = f"b{i:03d}"
    it["id"] = bid
    y, sr0 = librosa.load(it["src"], sr=None, mono=True)
    sf.write(OUT / "audio_blind" / f"{bid}.wav", y, sr0, subtype="PCM_16")
    y22, _ = librosa.load(it["src"], sr=SR, mono=True)
    it["duration"] = round(len(y22) / SR, 4)
    plot(y22, OUT / "img_blind" / f"{bid}.png")
(OUT / "blind.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

page = (OUT / "label.html").read_text(encoding="utf-8")
slim = [{"id": it["id"], "folder": "", "file": f"第 {n + 1} 个", "initial": "", "initial_kind": "", "start": 0,
         "end": it["duration"], "p": None, "duration": it["duration"]} for n, it in enumerate(items)]
start = page.index("const ROWS = ") + len("const ROWS = ")
end = page.index(";\n", start)
page = page[:start] + json.dumps(slim, ensure_ascii=False) + page[end:]
page = (page.replace("p_single", "p_blind").replace('"img2/"', '"img_blind/"').replace("img2/${r.id}", "img_blind/${r.id}")
        .replace('"audio/"+id', '"audio_blind/"+id')
        .replace("<title>P 点标注</title>", "<title>P 点盲标</title>")
        .replace("P 点标注：在频谱图上点出你认为正确的辅音/元音交界", "P 点盲标：这次不显示算法的线，请完全按耳朵点"))
import re as _re
page = page.replace('<p><b style="color:#ffd400">黄线</b> = 新算法的 P（黄色阴影是辅音段）；<b style="color:#3ef08a">绿线</b> = 你点的位置。点击频谱图任意处即可设置绿线，可反复点。</p>',
                    '<p>点击频谱图任意处设置<b style="color:#3ef08a">绿线</b>（你认为的辅音/元音交界），可反复点。这批里没有任何算法提示。</p>')
page = page.replace("黄线已经对了就点“黄线就对”。", "")
page = "\n".join(line for line in page.split("\n") if "▶ 从黄线" not in line and "黄线就对" not in line)
page = page.replace("　|　黄线与你的差：中位 ${med} ms，±15 ms 以内 ${within}/${offs.length}", "")
page = page.replace('off.textContent = r.p != null ? `你的点比黄线${L.p >= r.p ? "晚" : "早"} ${Math.abs((L.p - r.p)*1000).toFixed(0)} ms` : "";', 'off.textContent = "已标";')
(OUT / "label_blind.html").write_text(page, encoding="utf-8")
print(len(items), "items; repeats:", repeat_ids)
