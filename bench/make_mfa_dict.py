"""从 mandarin_mfa 汉字词典构造“无调拼音 → 音素”词典（每个拼音保留最常见的几种读法）。

在 otomad 环境运行（需要 pypinyin）。输出 bench/pinyin_mfa.dict，以及 phone 分类表 bench/mfa_initials.json：
MFA 音素是 IPA，声母（如 k, kʰ, ʈʂ, ɕ...）单列一个音素，韵母元音带声调符号。
"""
import collections
import json
import re
from pathlib import Path

from pypinyin import Style, pinyin

SRC = Path.home() / "Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict"
OUT = Path(__file__).resolve().parent
TONE = re.compile(r"[˥˦˧˨˩]")

prons = collections.defaultdict(collections.Counter)
for line in SRC.read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) < 6 or len(parts[0]) != 1 or not ("一" <= parts[0] <= "鿿"):
        continue
    ch, phones = parts[0], parts[5]
    readings = set(pinyin(ch, style=Style.NORMAL, heteronym=True)[0])
    if len(readings) == 1:  # 只用单一读音的汉字，避免多音字把读音配错
        prons[readings.pop().replace("ü", "v")][phones] += 1

lines, first_phones = [], collections.Counter()
for py, cnt in sorted(prons.items()):
    # 同一无调拼音的不同声调各保留一种（MFA 会自己挑最合适的）
    by_tone_free = {}
    for ph, n in cnt.most_common():
        key = TONE.sub("", ph)
        by_tone_free.setdefault(key, [])
        if len(by_tone_free[key]) < 4:
            by_tone_free[key].append(ph)
    for key, phs in by_tone_free.items():
        for ph in phs:
            lines.append(f"{py}\t{ph}")
            first_phones[ph.split()[0]] += 1
(OUT / "pinyin_mfa.dict").write_text("\n".join(lines) + "\n", encoding="utf-8")
initials = sorted(p for p in first_phones if not TONE.search(p) and len(p.split()) == 1)
(OUT / "mfa_initials.json").write_text(json.dumps(initials, ensure_ascii=False), encoding="utf-8")
print(len(prons), "pinyin;", len(lines), "entries; initial-like phones:", initials)
