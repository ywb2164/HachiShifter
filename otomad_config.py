"""otomad-agent 的路径配置：所有脚本从这里取路径，不再写死。

默认值按作者机器设置；换机器时用环境变量覆盖：
  OTOMAD_LIBRARY   素材库根目录（只读，永远不写入）           默认 D:\\杠氏老方
  OTOMAD_REPO      HachiShifter 源码仓库（含 build-integrated）  默认 D:\\HachiShifter\\HachiShifter-dev-latest-20261105
  OTOMAD_EXE       HachiShifter Next.exe                       默认 <OTOMAD_REPO>/build-integrated/.../RelWithDebInfo/HachiShifter Next.exe
  OTOMAD_MODELS    GAME/FCPE/NSF 模型目录（完整运行包里的 models） 默认 D:\\HachiShifter\\HachiShifter-0.27-Windows-x64-20261027\\models

本目录（otomad-agent）作为独立分支 `otomad-agent` 推送到 HachiShifter 的 fork；
HachiShifter 本身需要分支 `otomad-mcp-tools` 里新增的 MCP 工具。
"""
import os
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent

LIBRARY = Path(os.environ.get("OTOMAD_LIBRARY", r"D:\杠氏老方"))
REPO_ROOT = Path(os.environ.get("OTOMAD_REPO", r"D:\HachiShifter\HachiShifter-dev-latest-20261105"))
EXE = Path(os.environ.get("OTOMAD_EXE", REPO_ROOT / "build-integrated" / "HachiShifterNext_artefacts" / "RelWithDebInfo" / "HachiShifter Next.exe"))
MODELS = Path(os.environ.get("OTOMAD_MODELS", r"D:\HachiShifter\HachiShifter-0.27-Windows-x64-20261027\models"))

MIRROR = AGENT_ROOT / "library"                        # 素材硬链接镜像 + otto.ini（不入库）
ALIGNERS = AGENT_ROOT / "aligners"                     # HubertFA / SOFA 源码与模型（不入库）
HFA_DIR = ALIGNERS / "HubertFA"
HFA_MODEL = ALIGNERS / "hfa_model" / "1218_hfa_model_new_dict"
