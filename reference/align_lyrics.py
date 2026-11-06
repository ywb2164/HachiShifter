"""已知歌词（假名序列）+ 分离人声 → HubertFA 日语对齐 → 每个音拍的起点（元音起点 = UTAU 音符起点）。

测试用法：用 USTX 的歌词文字（不用其时间）在分离人声上对齐，再与 USTX 音符起点比较。
    python align_lyrics.py wonder/vocals.wav "<ustx>" --seconds 60
"""
import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "annotator"))
import otomad_config as cfg  # noqa: E402
from kana import to_romaji  # noqa: E402

VOWELISH = {"a", "i", "u", "e", "o", "N", "cl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("ustx")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    d = yaml.safe_load(open(a.ustx, encoding="utf-8-sig"))
    bpm, res = float(d["bpm"]), int(d.get("resolution", 480))
    p = d["voice_parts"][a.part]
    sec = lambda t: (p["position"] + t) / res * 60 / bpm  # noqa: E731
    notes = [n for n in sorted(p["notes"], key=lambda n: n["position"]) if sec(n["position"]) < a.seconds]
    off = json.load(open(HERE / "wonder" / "alignment.json"))["offset_seconds"]
    truth = [sec(n["position"]) + off for n in notes]
    roma = [to_romaji(n["lyric"]) for n in notes]

    # 只截取这一段音频（前后各留 0.5 s），按音频里 ≥0.35 s 的静音切成乐句，歌词按顺序分配给乐句由 HFA 一次完成
    y, sr = librosa.load(a.wav, sr=44100, mono=True)
    t0, t1 = max(0.0, truth[0] - 0.5), sec(notes[-1]["position"] + notes[-1]["duration"]) + off + 0.5
    seg = y[int(t0 * sr):int(t1 * sr)]
    work = HERE / "wonder" / "hfa"
    work.mkdir(exist_ok=True)
    sf.write(work / "seg.wav", seg, sr)
    (work / "seg.lab").write_text(" ".join(roma), encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(cfg.HFA_DIR)
    sys.path.insert(0, str(cfg.HFA_DIR))
    try:
        from onnx_infer import InferenceOnnx
        from tools.g2p import DictionaryG2P
        inf = InferenceOnnx(onnx_path=cfg.HFA_MODEL / "model.onnx")
        inf.load_config(); inf.init_decoder(); inf.load_model()
        g2p = DictionaryG2P("ja" if inf.vocab["language_prefix"] else None, cfg.HFA_MODEL / "japanese_dict_full.txt")
        ph, wd, idx = g2p(" ".join(roma))
        inf.dataset = [(work / "seg.wav", ph, wd, idx)]
        inf.predictions = []
        inf.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
    finally:
        os.chdir(cwd)
    words = [w for w in inf.predictions[0][2] if w.text not in {"SP", "AP", "EP", ""}]
    print(f"歌词 {len(roma)} 个音拍，对齐得到 {len(words)} 个")
    pred = []
    for w in words:
        phs = [x for x in w.phonemes if x.text not in {"SP", "AP", "EP", ""}]
        first = phs[0].text.split("/")[-1]
        onset = phs[0].start if (first in VOWELISH or len(phs) == 1) else phs[0].end  # 元音起点 = 音符起点
        pred.append(t0 + onset)
    n = min(len(pred), len(truth))
    err = np.array(pred[:n]) - np.array(truth[:n])
    ae = np.abs(err) * 1000
    print(f"音符起点误差（对齐 − USTX）：中位 {np.median(err) * 1000:+.0f} ms，|误差|中位 {np.median(ae):.0f} ms")
    print(f"  ≤30 ms {np.mean(ae <= 30):.0%}，≤60 ms {np.mean(ae <= 60):.0%}，≤100 ms {np.mean(ae <= 100):.0%}，>200 ms {int(np.sum(ae > 200))} 个")
    same = [i for i in range(1, n) if notes[i]["tone"] == notes[i - 1]["tone"]
            and abs(sec(notes[i - 1]["position"] + notes[i - 1]["duration"]) - sec(notes[i]["position"])) < 0.03]
    print(f"  同音高紧挨着的音符 {len(same)} 个：≤60 ms {np.mean(ae[same] <= 60):.0%}")
    json.dump({"onsets": pred, "romaji": roma}, open(work / "onsets.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
