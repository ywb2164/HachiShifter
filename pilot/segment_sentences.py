"""试点：整句素材按文件名台词自动切字，并给每个字找 P（辅音/元音交界）。

在 otomad 环境里运行（需要 funasr / pypinyin）：
    D:/miniconda/envs/otomad/python.exe segment_sentences.py [数量] [随机种子]

流程：
  1. 文件名 → 台词：去掉括号内容、编号、标点，只保留汉字；含英文字母的跳过
  2. FunASR fa-zh：已知文字的字级时间戳（对齐，不是听写）
  3. 每个字的范围内复用单字规则找 P（pyin 有声起点 + 声母类别）
  4. FunASR paraformer-zh 独立听写一遍，按拼音和文件名比对 → 每个字“ASR 是否认同”
  5. 自检规则标“自信 / 存疑”，输出 sentences.json、频谱图和试听审核页 sentences.html
素材库只读；切好的 16k 音频副本放在 pilot/sent_audio/。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import difflib
import json
import random
import re
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf
from pypinyin import Style, lazy_pinyin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Microsoft YaHei"

LIB = cfg.LIBRARY
OUT = Path(__file__).resolve().parent
FOLDERS = ["棍の音/水母卡组", "棍の音/电棍语音包", "剑魔语音+文字对照/文字语音", "棍の音/电棍原声大碟", "棍の音/山泥若语音包"]
SR = 22050
HOP = 110
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "r", "z", "c", "s", "y", "w"]
UNVOICED = {"b", "p", "d", "t", "g", "k", "j", "q", "x", "z", "c", "s", "zh", "ch", "sh", "f", "h"}
UNASPIRATED = {"b", "d", "g", "j", "z", "zh"}
SONORANT = {"m", "n", "l", "r"}


def text_from_name(stem: str) -> str | None:
    s = re.sub(r"[（(\[【].*?[)）\]】]", "", stem)
    if re.search(r"[A-Za-z]", s):
        return None
    han = "".join(re.findall(r"[\u4e00-\u9fff]", s))
    return han if len(han) >= 2 else None


def split_initial(py: str) -> tuple[str, str]:
    for ini in INITIALS:
        if py.startswith(ini) and len(py) > len(ini):
            return ini, "unvoiced" if ini in UNVOICED else "sonorant" if ini in SONORANT else "zero"
    return "", "zero"


def pick(n: int, seed: int) -> list[tuple[Path, str]]:
    rnd = random.Random(seed)
    pool = []
    for folder in FOLDERS:
        for f in sorted((LIB / folder).iterdir()):
            if f.suffix.lower() not in (".wav", ".mp3"):
                continue
            text = text_from_name(f.stem)
            if not text or len(text) > 16:
                continue
            try:
                dur = sf.info(f).duration if f.suffix.lower() == ".wav" else librosa.get_duration(path=f)
            except Exception:
                continue
            if 0.6 <= dur <= 8:
                pool.append((f, text))
    rnd.shuffle(pool)
    return pool[:n]


def first_run(mask, start, stop, length=4):
    run = 0
    for i in range(max(0, start), min(len(mask), stop)):
        run = run + 1 if mask[i] else 0
        if run >= length:
            return i - length + 1
    return None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    from funasr import AutoModel  # 延迟导入：加载较慢

    fa = AutoModel(model="fa-zh", disable_update=True)
    asr = AutoModel(model="paraformer-zh", disable_update=True)

    (OUT / "sent_audio").mkdir(exist_ok=True)
    (OUT / "sent_img").mkdir(exist_ok=True)
    out = []
    for k, (f, text) in enumerate(pick(n, seed)):
        sid = f"t{k:03d}"
        y16, _ = librosa.load(f, sr=16000, mono=True)
        wav16 = OUT / "sent_audio" / f"{sid}.wav"
        sf.write(wav16, y16, 16000, subtype="PCM_16")

        fa_res = fa.generate(input=(str(wav16), " ".join(text)), data_type=("sound", "text"))[0]
        stamps = fa_res.get("timestamp") or []
        asr_res = asr.generate(input=str(wav16))[0]
        asr_text = "".join(re.findall(r"[\u4e00-\u9fff]", asr_res.get("text", "")))

        pys = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
        if len(stamps) != len(text) or len(pys) != len(text):
            out.append({"id": sid, "file": f.name, "folder": str(f.parent.relative_to(LIB)), "text": text,
                        "asr_text": asr_text, "status": f"对齐失败（时间戳 {len(stamps)} 个 / 字 {len(text)} 个）", "chars": []})
            print(sid, f.name, "对齐失败")
            continue

        asr_py = lazy_pinyin(asr_text, style=Style.NORMAL, errors="ignore")
        sm = difflib.SequenceMatcher(a=pys, b=asr_py, autojunk=False)
        agree = [False] * len(pys)
        for a, _, size in sm.get_matching_blocks():
            for i in range(a, a + size):
                agree[i] = True
        ratio = sum(agree) / len(agree)

        y, _ = librosa.load(f, sr=SR, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1500, sr=SR, frame_length=1024, hop_length=HOP)
        rms = librosa.feature.rms(y=y, frame_length=441, hop_length=HOP)[0]
        db = 20 * np.log10(np.maximum(rms, 1e-6))
        nf = min(len(voiced), len(db))
        voiced = voiced[:nf].astype(bool)
        fr = lambda t: int(round(t * SR / HOP))  # noqa: E731
        tm = lambda i: round(i * HOP / SR, 4)  # noqa: E731

        chars = []
        prev_end = 0.0
        for i, (ch, py, (s_ms, e_ms)) in enumerate(zip(text, pys, stamps)):
            s, e = s_ms / 1000, e_ms / 1000
            ini, kind = split_initial(py)
            start = max(prev_end, s - 0.03)
            onset = first_run(voiced, fr(start), fr(e))
            reasons = []
            p = None
            if onset is None:
                reasons.append("范围内没有有声段")
            else:
                p_i = onset
                if kind == "sonorant":
                    voiced_idx = [j for j in range(onset, min(nf, fr(e))) if voiced[j]]
                    peak = db[onset:fr(e)].max() if fr(e) > onset else db[onset]
                    hit = [j for j in range(onset, min(nf, onset + fr(0.15), fr(e))) if db[j] >= peak - 6]
                    p_i = hit[0] if hit else onset + fr(0.06)
                    del voiced_idx
                p = tm(p_i)
                cons = (p - start) * 1000
                if kind == "unvoiced" and ini not in UNASPIRATED and cons < 15:
                    reasons.append("送气/擦音辅音段短于 15 ms")
                if cons > 300:
                    reasons.append("辅音段长于 300 ms")
                if (e - p) < 0.04:
                    reasons.append("元音段短于 40 ms")
            if e - s < 0.06:
                reasons.append("字长短于 60 ms")
            if not agree[i]:
                reasons.append("ASR 听写与文件名不一致")
            chars.append({"char": ch, "pinyin": py, "initial": ini, "initial_kind": kind,
                          "start": round(start, 4), "end": round(e, 4), "p": p,
                          "consonant_ms": None if p is None else round((p - start) * 1000, 1),
                          "asr_agree": agree[i], "verdict": "存疑" if reasons else "自信", "reasons": reasons})
            prev_end = e

        # 频谱图
        w = max(6.0, min(16.0, len(y) / SR * 2.2))
        fig, ax = plt.subplots(figsize=(w, 2.8), dpi=80)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)), ref=np.max)
        freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
        keep = freqs <= 8000
        ax.pcolormesh(np.arange(S.shape[1]) * HOP / SR, freqs[keep], S[keep], shading="auto", cmap="magma", vmin=-80, vmax=0)
        for c in chars:
            ax.axvline(c["start"], color="white", lw=1, ls="--")
            ax.axvline(c["end"], color="white", lw=0.6, ls=":")
            if c["p"] is not None:
                ax.axvline(c["p"], color="#ffd400", lw=1.5)
                ax.axvspan(c["start"], c["p"], color="#ffd400", alpha=0.15)
            ax.text((c["start"] + c["end"]) / 2, 7300, c["char"], color="white" if c["verdict"] == "自信" else "#ff8a65",
                    ha="center", fontsize=11)
        ax.set_ylim(0, 8000)
        ax.set_title(f"{sid}  {f.name}   ASR：{asr_text}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "sent_img" / f"{sid}.png")
        plt.close(fig)

        # 试听用：原采样率单声道副本
        mono, sr0 = librosa.load(f, sr=None, mono=True)
        sf.write(OUT / "sent_audio" / f"{sid}_play.wav", mono, sr0, subtype="PCM_16")

        status = "ok" if ratio >= 0.6 else "文件名与录音明显不符"
        out.append({"id": sid, "file": f.name, "folder": str(f.parent.relative_to(LIB)), "text": text,
                    "asr_text": asr_text, "asr_match": round(ratio, 2), "status": status, "chars": chars})
        conf = sum(c["verdict"] == "自信" for c in chars)
        print(sid, f.name, f"字 {len(chars)} 自信 {conf} ASR一致 {ratio:.0%}")

    (OUT / "sentences.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    allc = [c for s in out for c in s["chars"]]
    print(f"\n句子 {len(out)}，对齐失败 {sum(not s['chars'] for s in out)}，字 {len(allc)}，"
          f"自信 {sum(c['verdict'] == '自信' for c in allc)}，ASR 认同 {sum(c['asr_agree'] for c in allc)}")


if __name__ == "__main__":
    main()
