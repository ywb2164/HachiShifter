"""日语单字（单音拍）切片自动标注：文件名罗马音 → HubertFA 日语词典对齐 → otto.ini。

    D:/miniconda/envs/otomad/python.exe annotate_ja.py "日语人力素材/单字"

文件名形如 bi__12.wav（同一发音的第 12 个版本）、de1__0.wav、chan__0.wav。
- 去掉 __编号 和末尾数字得到发音；多音拍切片（chan/ai/shou…）按词典最长匹配拆成多个音节一起对齐
- 区域 = 能量包络覆盖整个发音；P = 第一个音素若是辅音则取其结束 +14 ms，否则（元音、ん开头）P = 起点
- 置信度：与规则法（低频能量）相差 ≤30 ms 记 0.9，否则 0.6；无法对齐的用规则法并记 0.4
- 别名：主名 = 发音（如 bi），附加别名 = 文件里的原写法（如 de1）与平假名
- annotations.json 额外记录 version（同发音的第几个版本）与源音高中位数，方便挑“移调最少”的版本
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate import (HFA_DIR, HFA_MODEL, LIB, MIRROR, P_OFFSET, EXE, energy_bounds, fl,  # noqa: E402
                      low_band_rise, mirror)
from kana import _BASE  # noqa: E402

VOWELISH = {"a", "i", "u", "e", "o", "N", "cl"}  # ん=N；小写 n 是な行的辅音
FIX = {"syi": "si", "zyi": "zi", "dyi": "di", "tyi": "ti"}
ROMA2KANA = {v: k for k, v in _BASE.items()}


def load_hfa_ja():
    cwd = os.getcwd()
    os.chdir(HFA_DIR)
    try:
        from onnx_infer import InferenceOnnx
        from tools.g2p import DictionaryG2P
        inf = InferenceOnnx(onnx_path=HFA_MODEL / "model.onnx")
        inf.load_config()
        inf.init_decoder()
        inf.load_model()
        g2p = DictionaryG2P("ja" if inf.vocab["language_prefix"] else None, HFA_MODEL / "japanese_dict_full.txt")
    finally:
        os.chdir(cwd)
    words = {}
    for line in (HFA_MODEL / "japanese_dict_full.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            w, ph = line.split("\t", 1)
            words[w] = ph.split()
    return inf, g2p, words


def split_tokens(name: str, words: dict) -> list[str] | None:
    """按词典最长匹配把 chan → [cha, n]、shou → [sho, u]。"""
    out, i = [], 0
    while i < len(name):
        for j in range(len(name), i, -1):
            if name[i:j] in words and name[i:j] not in {"SP", "AP"}:
                out.append(name[i:j])
                i = j
                break
        else:
            return None
    return out


def main() -> None:
    sub = sys.argv[1]
    src_root, dst_root = LIB / sub, MIRROR / sub
    files = sorted(p for p in src_root.rglob("*") if p.suffix.lower() in (".wav", ".mp3"))
    inf, g2p, words = load_hfa_ja()
    jobs, skipped = [], []
    for src in files:
        wav = mirror(src, dst_root / src.relative_to(src_root))
        m = re.match(r"^(.*?)(?:__(\d+))?$", src.stem)
        raw, version = m.group(1), m.group(2) or "0"
        base = re.sub(r"\d+$", "", raw.lower())
        name = FIX.get(base, base)
        tokens = split_tokens(name, words)
        job = {"wav": wav, "rel": str(src.relative_to(src_root)), "raw": raw, "name": name, "version": version, "tokens": tokens}
        if tokens:
            ph_seq, word_seq, idx = g2p(" ".join(tokens))
            job["hfa_in"] = (wav, ph_seq, word_seq, idx)
        jobs.append(job)
    todo = [j for j in jobs if "hfa_in" in j]
    print(f"{len(files)} 个文件，可对齐 {len(todo)}，词典里没有的 {len(jobs) - len(todo)}: {[j['raw'] for j in jobs if 'hfa_in' not in j]}")
    inf.dataset = [j["hfa_in"] for j in todo]
    inf.predictions = []
    cwd = os.getcwd()
    os.chdir(HFA_DIR)
    try:
        inf.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
    finally:
        os.chdir(cwd)
    pred = {str(p[0]): p[2] for p in inf.predictions}

    from mcp_client import HachiMcp
    mcp = HachiMcp(EXE, [f"--roots={MIRROR}"])
    index, written = [], 0
    for j in jobs:
        y, sr = librosa.load(j["wav"], sr=22050, mono=True)
        e_start, e_end = energy_bounds(y, sr)
        total = int(len(y) / sr * 10000) / 10000 - 0.0005
        notes = []
        phs = []
        for w in pred.get(str(j["wav"]), []):
            if w.text in {"SP", "AP", "EP", ""}:
                continue
            phs += [p for p in w.phonemes if p.text not in {"SP", "AP", "EP", ""}]
        start, end = e_start, min(e_end, total)
        rule = low_band_rise(y, sr, max(0.0, start - 0.03), end)
        if phs:
            first = phs[0].text.split("/")[-1]
            zero = first in VOWELISH or len(phs) == 1
            start, end = min(start, phs[0].start), min(max(end, phs[-1].end), total)
            p = start if zero else phs[0].end + P_OFFSET
            hfa_len = phs[-1].end - phs[0].start
            if hfa_len < 0.6 * (end - start):
                notes.append("HubertFA 字长明显短于实际发音")
                if rule is not None and not zero:
                    p = rule
                    notes.append("P 改用规则法")
            agree = zero or rule is None or abs(rule - p) <= 0.03
            conf = 0.9 if agree and not notes else 0.6
        else:
            zero = False
            p = rule if rule is not None else start
            conf = 0.4
            notes.append("词典里没有这个发音，P 用规则法")
        p = min(max(start, p), max(start, end - 0.03))
        f0med = None
        seg = y[int(p * sr):int(end * sr)]
        if len(seg) > 1024:
            f0, vf, _ = librosa.pyin(seg, fmin=80, fmax=1500, sr=sr, frame_length=1024, hop_length=220)
            if vf.any():
                f0med = round(float(librosa.hz_to_midi(np.nanmedian(f0[vf]))), 2)
        aliases = [a for a in dict.fromkeys([j["raw"], ROMA2KANA.get(j["name"], "")]) if a and a != j["name"]]
        rid = "p" + uuid.uuid4().hex
        row = {"name": j["name"], "region_start_seconds": fl(start), "region_end_seconds": fl(end),
               "alignment_seconds": fl(p), "fixed_duration_seconds": fl(p - start),
               "material_audio_id": "a" + uuid.uuid4().hex, "material_region_id": rid,
               "native_provenance": "otomad-hfa-ja", "native_confidence": round(conf, 2)}
        if aliases:
            row["additional_aliases"] = aliases
        ok, text = mcp.call("sample_settings_save", audio_path=str(j["wav"]), rows=[row])
        if not ok:
            skipped.append((j["rel"], text[:200]))
            continue
        written += 1
        index.append({"file": j["rel"], "name_in_library": Path(j["rel"]).name, "pinyin": j["name"], "char": "",
                      "romaji": j["name"], "version": j["version"], "source": "filename",
                      "start": round(start, 4), "end": round(end, 4), "p": round(p, 4),
                      "consonant_ms": round((p - start) * 1000, 1), "vowel_ms": round((end - p) * 1000, 1),
                      "median_midi": f0med, "confidence": round(conf, 2), "region_id": rid, "notes": notes,
                      "asr_text": "", "filename_not_heard": ""})
    mcp.close()
    (dst_root / "annotations.json").write_text(json.dumps({"rows": index, "skipped": skipped}, ensure_ascii=False, indent=1), encoding="utf-8")
    conf = [r["confidence"] for r in index]
    print(f"写入 {written} 个；置信度≥0.9 {sum(c >= 0.9 for c in conf)}，0.6 {sum(c == 0.6 for c in conf)}，≤0.4 {sum(c <= 0.4 for c in conf)}；"
          f"测到源音高 {sum(r['median_midi'] is not None for r in index)}；失败 {len(skipped)}")
    for s in skipped[:10]:
        print("  失败", s)


if __name__ == "__main__":
    main()
