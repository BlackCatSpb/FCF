"""
dashboard.py — real-time web dashboard for the autonomous think loop.

Serves:
  /          → HTML dashboard (auto-refreshing)
  /api/state → JSON current state from the think loop
  /api/log   → JSON recent log entries

Run: python -m eva.core.dashboard (standalone test)
Or imported by think_loop.py
"""
import os, json, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

PORT = 8383
_state = {}
_state_lock = threading.Lock()
_log_buffer = []
MAX_LOG = 200


def set_state(key, value):
    with _state_lock:
        _state[key] = value
        _state['updated_at'] = time.time()


def set_state_batch(d):
    with _state_lock:
        _state.update(d)
        _state['updated_at'] = time.time()


def get_state():
    with _state_lock:
        return dict(_state)


def log(msg: str):
    entry = {'t': time.strftime('%H:%M:%S'), 'msg': str(msg)}
    with _state_lock:
        _log_buffer.append(entry)
        if len(_log_buffer) > MAX_LOG:
            _log_buffer[:50] = []


def get_log(n=30):
    with _state_lock:
        return list(_log_buffer[-n:])


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>EVA — autonomous loop</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; font-size: 20px; margin-bottom: 16px; letter-spacing: 1px; }
h2 { color: #8b949e; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
.card .val { font-size: 28px; font-weight: 600; color: #f0f6fc; }
.card .label { font-size: 11px; color: #8b949e; margin-top: 2px; text-transform: uppercase; }
.card .bar { height: 6px; background: #21262d; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.card .bar-fill { height: 100%; background: #58a6ff; border-radius: 3px; transition: width 1s; }
.phase { font-size: 16px; font-weight: 600; padding: 4px 12px; border-radius: 12px; display: inline-block; }
.phase-THINK { background: #1f6feb33; color: #58a6ff; }
.phase-ANALYZE { background: #23863633; color: #3fb950; }
.phase-LEARN { background: #9e6a0333; color: #d29922; }
.phase-OPTIMIZE { background: #da363333; color: #f85149; }
.phase-IDLE { background: #21262d; color: #8b949e; }
.log { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.6; max-height: 300px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace; }
.log-entry { color: #8b949e; }
.log-entry .time { color: #484f58; }
.bar-chart { display: flex; gap: 2px; align-items: flex-end; height: 60px; margin-top: 6px; }
.bar-item { flex: 1; border-radius: 2px 2px 0 0; min-height: 2px; position: relative; }
.bar-label { font-size: 9px; text-align: center; color: #8b949e; margin-top: 2px; }
.row { display: flex; gap: 20px; flex-wrap: wrap; }
.col { flex: 1; min-width: 280px; }
#timestamp { font-size: 11px; color: #484f58; margin-bottom: 12px; }
</style>
</head>
<body>
<h1>⬡ EVA — autonomous loop</h1>
<div id="timestamp"></div>
<div class="grid" id="cards"></div>
<div class="row">
<div class="col">
  <h2>Heads weight distribution</h2>
  <div class="card"><div class="bar-chart" id="head-chart"></div></div>
</div>
<div class="col">
  <h2>Accuracy trend (last 100)</h2>
  <div class="card"><div class="bar-chart" id="acc-chart"></div></div>
</div>
</div>
<div class="row">
<div class="col">
  <h2>Generation rate trend (tok/s)</h2>
  <div class="card"><div class="bar-chart" id="rate-chart"></div></div>
</div>
<div class="col"></div>
</div>
<h2>Event log</h2>
<div class="log" id="log"></div>
<script>
async function fetchJSON(url) {
  try { const r = await fetch(url); return await r.json(); }
  catch { return null; }
}
function render(state) {
  if (!state) return;
  const cards = document.getElementById('cards');
  const phase = state.phase || 'IDLE';
  const phaseLabel = `<span class="phase phase-${phase}">${phase}</span>`;
  
  cards.innerHTML = [
    { label: 'Phase', val: phaseLabel },
    { label: 'Tokens generated', val: state.tokens_generated || 0 },
    { label: 'Sentences', val: state.db_sentences || 0 },
    { label: 'Contradictions', val: state.contradictions || 0 },
    { label: 'Concepts', val: state.concepts || 0 },
    { label: 'Transformer accuracy', val: (state.transformer_acc || 0).toFixed(1) + '%' },
    { label: 'Generation rate', val: (state.gen_rate || 0).toFixed(1) + ' tok/s' },
    { label: 'Uptime', val: state.uptime || '0s' },
    { label: 'Disk usage', val: state.disk_usage || '0 MB' },
  ].map(c => `
    <div class="card">
      <div class="val">${c.val}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');

  document.getElementById('timestamp').textContent =
    'updated: ' + (state.updated_at || '');

  // Head weight bar chart
  if (state.head_weights) {
    const hw = state.head_weights;
    const names = ['morph','syntax','trans','sem','concept','contra'];
    const maxW = Math.max(...hw, 0.1);
    document.getElementById('head-chart').innerHTML = hw.map((v, i) => `
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;">
        <div class="bar-item" style="background:#58a6ff;height:${(v/maxW)*55}px;" title="${names[i]}: ${v.toFixed(2)}"></div>
        <div class="bar-label">${names[i]}</div>
      </div>
    `).join('');
  }

  // Accuracy trend (last 100)
  if (state.acc_history && state.acc_history.length > 0) {
    const accs = state.acc_history;
    const maxA = Math.max(...accs, 1);
    document.getElementById('acc-chart').innerHTML = accs.map((v, i) => `
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;">
        <div class="bar-item" style="background:${v > 0.1 ? '#3fb950' : '#21262d'};height:${(v/maxA)*55}px;" title="${(v*100).toFixed(1)}%"></div>
      </div>
    `).join('');
  }

  // Generation rate trend (last 100)
  if (state.rate_history && state.rate_history.length > 0) {
    const rates = state.rate_history;
    const maxR = Math.max(...rates, 1);
    document.getElementById('rate-chart').innerHTML = rates.map((v, i) => `
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;">
        <div class="bar-item" style="background:#f0883e;height:${(v/maxR)*55}px;" title="${v.toFixed(0)} tok/s"></div>
      </div>
    `).join('');
  }
}

async function update() {
  const state = await fetchJSON('/api/state');
  render(state);
  const logData = await fetchJSON('/api/log');
  if (logData) {
    document.getElementById('log').innerHTML = logData.map(e =>
      `<div class="log-entry"><span class="time">[${e.t}]</span> ${e.msg}</div>`
    ).join('');
    const logEl = document.getElementById('log');
    logEl.scrollTop = logEl.scrollHeight;
  }
}
setInterval(update, 2500);
update();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/state':
            self._json(get_state())
        elif self.path == '/api/log':
            self._json(get_log())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))

    def _json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

    def log_message(self, *a):
        pass  # silence


def start_server(port=PORT):
    server = HTTPServer(('127.0.0.1', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f'Dashboard at http://127.0.0.1:{port}')
    return server


if __name__ == '__main__':
    print(f'Dashboard at http://127.0.0.1:{PORT}')
    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
