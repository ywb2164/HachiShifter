"""试点用本地服务器：提供静态文件，并接收网页提交的人工标注（只监听 127.0.0.1）。

GET  /labels?set=<名字>   读取已保存的标注
POST /labels?set=<名字>   覆盖保存（JSON），写到 labels_<名字>.json
"""
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def _target(self):
        q = parse_qs(urlparse(self.path).query)
        name = re.sub(r"[^A-Za-z0-9_-]", "", (q.get("set") or ["default"])[0]) or "default"
        return ROOT / f"labels_{name}.json"

    def do_GET(self):
        if urlparse(self.path).path == "/labels":
            f = self._target()
            body = f.read_bytes() if f.exists() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/labels":
            self.send_error(404)
            return
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        json.loads(data)  # 校验
        tmp = self._target().with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(self._target())
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
