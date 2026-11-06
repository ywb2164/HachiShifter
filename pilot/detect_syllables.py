"""试点：单字素材的辅音/元音交界（P）自动检测。

不改动素材库：音频先复制成单声道 WAV 放到 pilot/audio/，结果写入 pilot/results.json，
每个素材生成一张频谱图 pilot/img/<id>.png。

判定规则（全部是信号层面的事实，不靠模型猜）：
  - 起止：帧能量高于峰值 -40 dB 的第一帧 / 最后一帧
  - 有声起点：pyin 连续 4 帧（20 ms）判为有声的第一帧
  - 清辅音声母（b p d t g k j q x z c s zh ch sh f h）：P = 有声起点，辅音段 = 起点..P
  - 浊辅音声母（m n l r）：P = 有声段内能量首次接近元音峰值（-6 dB）处，最多有声起点后 150 ms
  - 零声母（a o e y w ...）：P = 有声起点
  - 自检：清辅音段过短/过长、零声母前面有长噪声、找不到有声段 → 标“存疑”
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import json
import random
import re
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LIB = cfg.LIBRARY
OUT = Path(__file__).resolve().parent
SR = 22050
HOP = 110  # ≈5 ms

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "r", "z", "c", "s", "y", "w"]
UNVOICED = {"b", "p", "d", "t", "g", "k", "j", "q", "x", "z", "c", "s", "zh", "ch", "sh", "f", "h"}
UNASPIRATED = {"b", "d", "g", "j", "z", "zh"}  # 不送气塞音/塞擦音，辅音段本来就很短
SONORANT = {"m", "n", "l", "r"}


def split_pinyin(name: str) -> tuple[str, str, str]:
    syl = re.sub(r"[^a-zv]", "", name.lower())
    for ini in INITIALS:
        if syl.startswith(ini) and len(syl) > len(ini):
            kind = "unvoiced" if ini in UNVOICED else "sonorant" if ini in SONORANT else "zero"
            return syl, ini, kind
    return syl, "", "zero"


def first_run(mask: np.ndarray, start: int, length: int) -> int | None:
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= length:
            return i - length + 1
    return None


def analyse(path: Path) -> dict:
    y, _ = librosa.load(path, sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=441, hop_length=HOP, center=True)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    peak = db.max()
    above = np.where((db > peak - 40) & (db > -60))[0]
    if len(above) == 0:
        return {"status": "silent"}
    start, end = int(above[0]), int(above[-1])

    f0, voiced, prob = librosa.pyin(y, fmin=65, fmax=1500, sr=SR, frame_length=1024,
                                    hop_length=HOP, center=True)
    n = min(len(db), len(voiced))
    voiced = voiced[:n].astype(bool)
    onset = first_run(voiced, start, 4)
    t = lambda i: round(i * HOP / SR, 4)  # noqa: E731
    base = {"start": t(start), "end": t(end), "f0": [None if np.isnan(v) else round(float(v), 1) for v in f0[:n]],
            "db": [round(float(v), 1) for v in db[:n]]}
    if onset is None:
        return {**base, "status": "unvoiced_only"}
    voiced_idx = np.where(voiced)[0]
    voice_end = int(voiced_idx[-1])
    vowel_peak = float(db[onset:voice_end + 1].max())
    med_f0 = float(np.nanmedian(f0[onset:voice_end + 1]))
    return {**base, "status": "ok", "onset_i": onset, "voice_onset": t(onset),
            "voice_end": t(voice_end), "vowel_peak_db": round(vowel_peak, 1),
            "median_f0": round(med_f0, 1), "median_midi": round(float(librosa.hz_to_midi(med_f0)), 2)}


def place_p(r: dict, kind: str, ini: str = "") -> None:
    if r["status"] != "ok":
        r.update(p=None, consonant_ms=None, verdict="存疑", reasons=[{"silent": "近乎无声", "unvoiced_only": "找不到有声段（可能是气声或超高音鬼叫）"}[r["status"]]])
        return
    onset = r["onset_i"]
    db = np.array(r["db"])
    p = onset
    if kind == "sonorant":
        lim = min(len(db) - 1, onset + int(0.150 * SR / HOP))
        hit = np.where(db[onset:lim + 1] >= r["vowel_peak_db"] - 6)[0]
        p = onset + int(hit[0]) if len(hit) else onset + int(0.06 * SR / HOP)
    r["p"] = round(p * HOP / SR, 4)
    cons = (r["p"] - r["start"]) * 1000
    r["consonant_ms"] = round(cons, 1)
    reasons = []
    if kind == "unvoiced" and ini not in UNASPIRATED and cons < 15:
        reasons.append("送气/擦音辅音段短于 15 ms")
    if kind == "unvoiced" and cons > 300:
        reasons.append("辅音段长于 300 ms")
    if kind == "zero" and cons > 80:
        reasons.append("零声母但起点前有 80 ms 以上的声音")
    if (r["voice_end"] - r["p"]) < 0.04:
        reasons.append("元音段短于 40 ms")
    r["verdict"] = "存疑" if reasons else "自信"
    r["reasons"] = reasons


def plot(y: np.ndarray, r: dict, title: str, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=90)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)), ref=np.max)
    times = np.arange(S.shape[1]) * HOP / SR
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    keep = freqs <= 8000
    ax.pcolormesh(times, freqs[keep], S[keep], shading="auto", cmap="magma", vmin=-80, vmax=0)
    f0 = np.array([np.nan if v is None else v for v in r.get("f0", [])], dtype=float)
    if len(f0):
        ax.plot(np.arange(len(f0)) * HOP / SR, f0, color="#4fd1ff", lw=1.4)
    for key, col, lab in (("start", "white", "起"), ("p", "#ffd400", "P"), ("end", "white", "止")):
        if r.get(key) is not None:
            ax.axvline(r[key], color=col, lw=1.6, ls="-" if key == "p" else "--")
    if r.get("p") is not None:
        ax.axvspan(r["start"], r["p"], color="#ffd400", alpha=0.18)
    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    ax.set_xlabel("s")
    ax.set_title(title, fontsize=9, fontfamily="Microsoft YaHei")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def main() -> None:
    random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
    picks = []
    for folder, n in (("棍の音/棍音源", 40), ("张顺飞素材/张顺飞活字印刷", 40)):
        files = sorted(p for p in (LIB / folder).iterdir() if p.suffix.lower() == ".wav")
        picks += [(folder, f) for f in random.sample(files, n)]
    (OUT / "audio").mkdir(exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)
    rows = []
    for k, (folder, f) in enumerate(picks):
        rid = f"s{k:03d}"
        syl, ini, kind = split_pinyin(f.stem)
        y_full, sr0 = sf.read(f, always_2d=True)
        mono = y_full.mean(axis=1)
        sf.write(OUT / "audio" / f"{rid}.wav", mono, sr0, subtype="PCM_16")
        r = analyse(f)
        place_p(r, kind, ini)
        y, _ = librosa.load(f, sr=SR, mono=True)
        plot(y, r, f"{rid}  {f.name}  声母={ini or '零'}  {r['verdict']}", OUT / "img" / f"{rid}.png")
        r.pop("onset_i", None)
        r.pop("f0", None)
        r.pop("db", None)
        rows.append({"id": rid, "folder": folder, "file": f.name, "syllable": syl, "initial": ini,
                     "initial_kind": kind, **r})
        print(rid, f.name, ini or "-", r.get("p"), r.get("consonant_ms"), r["verdict"], ";".join(r["reasons"]))
    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    conf = sum(r["verdict"] == "自信" for r in rows)
    print(f"\n共 {len(rows)} 个，自信 {conf}，存疑 {len(rows) - conf}")


if __name__ == "__main__":
    main()
