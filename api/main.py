from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


class PrototypeApi(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        payload_path = OUTPUTS / "api_payload.json"
        if not payload_path.exists():
            self._send_json({"error": "Demo outputs not generated. Run run_demo.py first."}, 404)
            return

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        routes = {
            "/": {"service": "cimento-mpc-prototype", "routes": ["/summary", "/metrics", "/quality", "/recommendations", "/economics"]},
            "/summary": payload.get("summary", {}),
            "/metrics": payload.get("metrics", []),
            "/quality": {
                "metrics": payload.get("quality_metrics", []),
                "predictions": payload.get("quality_predictions", []),
            },
            "/recommendations": payload.get("recommendations", []),
            "/economics": payload.get("economics", []),
        }
        self._send_json(routes.get(path, {"error": "not found"}), 200 if path in routes else 404)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), PrototypeApi)
    print(f"API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
