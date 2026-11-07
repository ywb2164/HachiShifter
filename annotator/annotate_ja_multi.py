"""日语多字（多音拍词/短句）切片标注：每个音拍一个区域，区域首尾相接，供 material_query 找“同一条录音里连续说出”的链。

    D:/miniconda/envs/otomad/python.exe annotate_ja_multi.py "日语人力素材/多字（原声大耋）"

文件名形如 hi_to_ri.wav、mi_N_na.wav、i_tsu_ka1.wav（末尾数字/ _1 是第几个版本）。
- 下划线分段，N=ん，sh=し，k=く（无声化），段内按词典最长匹配再拆（hon→ho+n，rai→ra+i）
- HubertFA 日语词典对齐；第 k 个音拍区域 = [它的第一个音素起点, 下一个音拍起点)，首拍从能量起点、末拍到能量终点
- 同一文件所有音拍共用一个 material_audio_id，并写 connected_next_region_id，HachiShifter 才认作连续链
- P = 首音素若是辅音取其结束 +14 ms，否则 = 区域起点（与单字标注一致）
- 原始素材库只读，只写 otomad-agent/library 镜像；annotations.json 每个音拍一行，带 chain/chain_pos/median_midi
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
from annotate import HFA_DIR, LIB, MIRROR, P_OFFSET, EXE, energy_bounds, fl, mirror  # noqa: E402
from annotate_ja import load_hfa_ja, split_tokens, VOWELISH, FIX, ROMA2KANA  # noqa: E402

PART_MAP = {"N": "n", "n": "n", "sh": "shi", "k": "ku", "t": "tsu"}
KANA_FIX = {"n": "ん"}


def tokens_for(stem: str, words: dict) -> tuple[list[str], str] | None:
    parts = stem.split("_")
    version = "0"
    if len(parts) > 1 and parts[-1].isdigit():
        version = parts.pop()
    m = re.match(r"^(.*?)(\d+)$", parts[-1])
    if m and m.group(1):
        parts[-1], version = m.group(1), m.group(2)
    out = []
    for p in parts:
        key = PART_MAP.get(p, p.lower())
        key = FIX.get(key, key)
        sub = [key] if key in words else split_tokens(key, words)
        if not sub:
            return None
        out += sub
    return out, version


def kana(tok: str) -> str:
    return KANA_FIX.get(tok) or ROMA2KANA.get(tok, "")


def main() -> None:
    sub = sys.argv[1]
    src_root, dst_root = LIB / sub, MIRROR / sub
    files = sorted(p for p in src_root.rglob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac"))
    inf, g2p, words = load_hfa_ja()
    jobs, skipped = [], []
    for src in files:
        parsed = tokens_for(src.stem, words)
        if not parsed or len(parsed[0]) < 1:
            skipped.append((src.name, "无法解析文件名为音拍"))
            continue
        toks, version = parsed
        wav = mirror(src, dst_root / src.relative_to(src_root))
        ph_seq, word_seq, idx = g2p(" ".join(toks))
        jobs.append({"wav": wav, "rel": str(src.relative_to(src_root)), "stem": src.stem, "tokens": toks, "version": version,
                     "hfa_in": (wav, ph_seq, word_seq, idx)})
    print(f"{len(files)} 个文件，可解析 {len(jobs)}；跳过 {skipped}")
    inf.dataset = [j["hfa_in"] for j in jobs]
    inf.predictions = []
    cwd = os.getcwd(); os.chdir(HFA_DIR)
    try:
        inf.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
    finally:
        os.chdir(cwd)
    pred = {str(p[0]): p[2] for p in inf.predictions}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from mcp_client import HachiMcp
    mcp = HachiMcp(EXE, [f"--roots={MIRROR}"])
    index, written = [], 0
    for j in jobs:
        y, sr = librosa.load(j["wav"], sr=22050, mono=True)
        total = int(len(y) / sr * 10000) / 10000 - 0.0005
        e_start, e_end = energy_bounds(y, sr)
        ws = [w for w in pred.get(str(j["wav"]), []) if w.text not in {"SP", "AP", "EP", ""}]
        if len(ws) != len(j["tokens"]):
            skipped.append((j["rel"], f"对齐出 {len(ws)} 个词，期望 {len(j['tokens'])}"))
            continue
        starts = [w.phonemes[0].start for w in ws]
        starts[0] = min(starts[0], e_start)
        ends = starts[1:] + [min(max(e_end, ws[-1].phonemes[-1].end), total)]
        audio_id = "a" + uuid.uuid4().hex
        rids = ["p" + uuid.uuid4().hex for _ in ws]
        rows, meta = [], []
        for k, (w, tok) in enumerate(zip(ws, j["tokens"])):
            a, b = fl(starts[k]), fl(ends[k])
            if b <= a + 0.02:
                break
            first = w.phonemes[0].text.split("/")[-1]
            zero = first in VOWELISH or len(w.phonemes) == 1
            p = a if zero else w.phonemes[0].end + P_OFFSET
            p = fl(min(max(a, p), b - 0.02))
            f0med = None
            seg = y[int(p * sr):int(b * sr)]
            if len(seg) > 1024:
                f0, vf, _ = librosa.pyin(seg, fmin=80, fmax=1500, sr=sr, frame_length=1024, hop_length=220)
                if vf.any():
                    f0med = round(float(librosa.hz_to_midi(np.nanmedian(f0[vf]))), 2)
            row = {"name": tok, "region_start_seconds": a, "region_end_seconds": b, "alignment_seconds": p,
                   "fixed_duration_seconds": fl(p - a), "material_audio_id": audio_id, "material_region_id": rids[k],
                   "native_provenance": "otomad-hfa-ja-multi", "native_confidence": 0.8}
            if kana(tok):
                row["additional_aliases"] = [kana(tok)]
            if k + 1 < len(ws):
                row["connected_next_region_id"] = rids[k + 1]
            rows.append(row)
            meta.append({"file": j["rel"], "name_in_library": Path(j["rel"]).name, "pinyin": tok, "char": kana(tok),
                         "romaji": tok, "version": j["version"], "source": "filename-multi", "chain": j["stem"],
                         "chain_pos": k, "chain_len": len(ws), "start": round(a, 4), "end": round(b, 4), "p": round(p, 4),
                         "consonant_ms": round((p - a) * 1000, 1), "vowel_ms": round((b - p) * 1000, 1),
                         "median_midi": f0med, "confidence": 0.8, "region_id": rids[k], "notes": [],
                         "asr_text": " ".join(j["tokens"]), "filename_not_heard": ""})
        if rows and "connected_next_region_id" in rows[-1]:
            rows[-1].pop("connected_next_region_id")
        ok, text = mcp.call("sample_settings_save", audio_path=str(j["wav"]), rows=rows)
        if not ok:
            skipped.append((j["rel"], text[:200]))
            continue
        written += 1
        index += meta
    mcp.close()
    (dst_root / "annotations.json").write_text(json.dumps({"rows": index, "skipped": skipped}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {written} 个文件，{len(index)} 个音拍区域；测到源音高 {sum(r['median_midi'] is not None for r in index)}；失败/跳过 {len(skipped)}")
    for s in skipped[:15]:
        print("  ", s)


if __name__ == "__main__":
    main()
