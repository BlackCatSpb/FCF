"""Serve 3D viewer on local network.
Usage:  python serve_vis.py [port]

Open http://<your-ip>:8080/viewer.html from any device.
"""
import sys, os, http.server, socketserver, socket

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
DIR = os.path.join(os.path.dirname(__file__), 'real_data', 'vis')

if not os.path.isdir(DIR):
    print(f"vis directory not found: {DIR}")
    sys.exit(1)

# Detect local IP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('10.255.255.255', 1))
    IP = s.getsockname()[0]
except Exception:
    IP = '127.0.0.1'
finally:
    s.close()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

print(f"Serving {DIR}")
print(f"  Local:   http://127.0.0.1:{PORT}/viewer.html")
print(f"  Network: http://{IP}:{PORT}/viewer.html")
with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    httpd.serve_forever()
