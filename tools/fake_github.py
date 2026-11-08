# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
import json, sys
# Test helper, not part of the program: a pretend GitHub release for the updater.
# Usage: fake_github.py <port> <version> <LoraStudio.zip>
from http.server import BaseHTTPRequestHandler, HTTPServer
port, tag, zipf = int(sys.argv[1]), sys.argv[2], sys.argv[3]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/releases/latest"):
            body = json.dumps({"tag_name": "v" + tag, "body": "Test release notes:\n- one\n- two", "html_url": "https://github.com/x/y/releases/tag/v" + tag,
                               "assets": [{"name": "LoraStudio.zip", "browser_download_url": f"http://127.0.0.1:{port}/LoraStudio.zip"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        elif self.path == "/LoraStudio.zip":
            data = open(zipf, "rb").read()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
