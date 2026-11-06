"""最小 MCP stdio 客户端：启动 HachiShifter --mcp，逐条调用工具。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class HachiMcp:
    def __init__(self, exe: str | Path, extra_args: list[str] | None = None):
        self.proc = subprocess.Popen([str(exe), "--mcp", *(extra_args or [])], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._id = 0
        self.request("initialize", {})

    def request(self, method: str, params: dict) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server exited")
            reply = json.loads(line.decode("utf-8"))
            if reply.get("id") == self._id:
                return reply

    def call(self, name: str, /, **arguments) -> tuple[bool, str]:
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        if "error" in reply:
            return False, json.dumps(reply["error"], ensure_ascii=False)
        res = reply["result"]
        text = "".join(c.get("text", "") for c in res.get("content", []))
        return not res.get("isError", False), text

    def call_json(self, name: str, /, **arguments):
        ok, text = self.call(name, **arguments)
        if not ok:
            raise RuntimeError(f"{name} failed: {text}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def tools(self) -> list[str]:
        return [t["name"] for t in self.request("tools/list", {})["result"]["tools"]]

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
