"""
=============================================================================
CADASTRAL PANORAMA BUILDER  —  FastAPI Web UI
=============================================================================
Run from the thesis root directory:
    python new_pipeline/src/panorama_api.py

Then open in your browser:
    http://localhost:8000
=============================================================================
"""

import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

# ── working directory = thesis root ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).parent))

from panorama_all import run as run_panorama, MAPS, OUT_PATH
from edge_orientation_finder import (
    PREPROCESSED_DIR, RESULTS_DIR, ADJACENT_PAIRS,
    auto_crop, read_image, analyse_pair,
)

# ── app ──────────────────────────────────────────────────────────────────────
app   = FastAPI()
_jobs = {}   # job_id -> { queue, result_path }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML.replace("__MAPS__", json.dumps(MAPS)))


@app.get("/thumb/{map_num}")
async def thumbnail(map_num: str):
    if map_num not in MAPS:
        return Response(status_code=404)
    path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    if not path.exists():
        return Response(status_code=404)
    try:
        img   = auto_crop(read_image(path))
        h, w  = img.shape
        scale = min(200 / w, 260 / h)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception:
        return Response(status_code=500)


@app.post("/generate")
async def generate(body: dict):
    job_id = str(uuid.uuid4())[:8]
    queue  = asyncio.Queue()
    loop   = asyncio.get_running_loop()
    _jobs[job_id] = {"queue": queue, "result_path": None}
    threading.Thread(
        target=_run_job,
        args=(job_id, body["maps"], loop, queue),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.websocket("/ws/{job_id}")
async def ws_endpoint(ws: WebSocket, job_id: str):
    await ws.accept()
    job = _jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "text": "Job not found"})
        return
    try:
        while True:
            msg = await asyncio.wait_for(job["queue"].get(), timeout=600.0)
            await ws.send_json(msg)
            if msg["type"] in ("done", "error"):
                break
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/result/{job_id}")
async def result(job_id: str):
    job  = _jobs.get(job_id)
    path = job and job.get("result_path") and Path(job["result_path"])
    if not path or not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/png",
                    headers={"Content-Disposition": "inline; filename=panorama.png"})


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_job(job_id: str, selected_maps: list, loop, queue):
    def send(msg):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

    class _Stdout:
        def write(self, t):
            if t:
                send({"type": "log", "text": t})
        def flush(self): pass

    old_out    = sys.stdout
    sys.stdout = _Stdout()

    try:
        sel = set(selected_maps)

        # ── Step 1: edge orientation ─────────────────────────────────────
        send({"type": "step", "step": 1})
        pairs = [(a, b) for a, b in ADJACENT_PAIRS if a in sel and b in sel]
        n     = max(len(pairs), 1)

        for i, (a, b) in enumerate(pairs):
            jpath = RESULTS_DIR / f"pair_{a}_{b}_scores.json"
            if not jpath.exists():
                send({"type": "log",
                      "text": f"\n  Analysing pair {a} ↔ {b} …\n"})
                analyse_pair(a, b, visualise=False)
            send({"type": "progress", "value": 0.05 + 0.60 * (i + 1) / n})

        # ── Step 2: panorama assembly ────────────────────────────────────
        send({"type": "step", "step": 2})
        exclude  = [m for m in MAPS if m not in sel]
        out_path = RESULTS_DIR / f"panorama_{job_id}.png"
        run_panorama(exclude=exclude, out_path=out_path)

        sys.stdout = old_out
        _jobs[job_id]["result_path"] = str(out_path)
        send({"type": "progress", "value": 1.0})
        send({"type": "step",    "step": 3})
        send({"type": "done",    "job_id": job_id})

    except Exception:
        import traceback
        sys.stdout = old_out
        send({"type": "error", "text": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# Embedded single-page frontend
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cadastral Panorama Builder</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0b0b; --surf:#111; --card:#161616; --card-on:#0b2117;
  --bdr:#1f1f1f; --bdr-on:#4ade80;
  --green:#4ade80; --gd:#163724; --gdd:#0b2117;
  --blue:#60a5fa; --red:#f87171;
  --fg:#e0e0e0; --fdim:#5a5a5a; --fmid:#909090;
  --r:12px;
}
html{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--fg);font-size:14px}
body{min-height:100vh;display:flex;flex-direction:column}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:3px}

/* ── header ── */
header{
  background:var(--surf);border-bottom:1px solid var(--bdr);
  padding:16px 30px;display:flex;align-items:center;gap:14px;flex-shrink:0;
}
.logo{
  width:42px;height:42px;border-radius:10px;background:var(--gd);
  display:flex;align-items:center;justify-content:center;font-size:21px;flex-shrink:0;
}
.htitle{font-size:17px;font-weight:700;color:var(--green);letter-spacing:.3px}
.hsub{font-size:11px;color:var(--fdim);margin-top:2px}

/* ── main ── */
main{flex:1;padding:22px 30px;max-width:1080px;margin:0 auto;width:100%}

/* ── toolbar ── */
.toolbar{
  display:flex;align-items:center;gap:14px;
  background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);
  padding:11px 18px;margin-bottom:18px;
}
.selall{display:flex;align-items:center;gap:9px;cursor:pointer;user-select:none;font-size:13px;font-weight:600}
.chk{
  width:19px;height:19px;border:2px solid var(--bdr-on);border-radius:5px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;transition:background .15s;
}
.selall.on .chk{background:var(--green)}
.selall.on .chk::after{content:'✓';color:#000;font-size:11px;font-weight:900}
.badge{font-size:12px;color:var(--fdim);background:var(--bdr);border-radius:20px;padding:3px 12px}
.spc{flex:1}
.btn-gen{
  background:var(--gd);color:var(--green);border:1.5px solid var(--green);
  border-radius:8px;padding:9px 28px;font-size:13px;font-weight:700;letter-spacing:.4px;
  cursor:pointer;transition:background .18s;
}
.btn-gen:hover:not(:disabled){background:#1f5e32}
.btn-gen:disabled{opacity:.35;cursor:not-allowed}

/* ── map grid ── */
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}
.card{
  background:var(--card);border:2px solid var(--bdr);border-radius:var(--r);
  padding:10px;cursor:pointer;user-select:none;
  transition:border-color .16s,background .16s;
}
.card:hover{border-color:#2e2e2e;background:#1b1b1b}
.card.on{background:var(--card-on);border-color:var(--bdr-on)}
.thumb{
  width:100%;aspect-ratio:4/5;background:#0e0e0e;border-radius:8px;
  overflow:hidden;display:flex;align-items:center;justify-content:center;margin-bottom:9px;
}
.thumb img{width:100%;height:100%;object-fit:contain;display:block}
.thumb .nop{font-size:11px;color:var(--fdim)}
.cfoot{display:flex;align-items:center;justify-content:space-between;padding:0 2px}
.mlbl{font-size:13px;font-weight:700}
.dot{width:17px;height:17px;border-radius:50%;border:2px solid #2a2a2a;transition:all .15s}
.card.on .dot{background:var(--green);border-color:var(--green)}

/* ── progress panel ── */
.ppanel{
  background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);
  padding:20px 22px;margin-bottom:18px;display:none;
}
.ppanel.show{display:block}

.steps{display:flex;align-items:center;margin-bottom:18px}
.step{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:600;color:var(--fdim);transition:color .3s}
.step.active{color:var(--green)}
.step.done{color:#444}
.sc{
  width:28px;height:28px;border-radius:50%;border:2px solid currentColor;
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;
  flex-shrink:0;transition:background .3s;position:relative;
}
.step.active .sc{background:var(--gd)}
.step.done .sc{background:#181818;border-color:#333;font-size:0}
.step.done .sc::after{content:'✓';font-size:13px;color:#555;position:absolute}
.sln{flex:1;height:1px;background:var(--bdr);margin:0 10px;max-width:64px}
.sln.done{background:#363636}

@keyframes pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(74,222,128,.3)}
  50%{box-shadow:0 0 0 8px rgba(74,222,128,0)}
}
.step.active .sc{animation:pulse 1.7s infinite}

.track{height:3px;background:var(--bdr);border-radius:2px;margin-bottom:16px;overflow:hidden}
.bar{height:100%;background:var(--green);border-radius:2px;width:0;transition:width .35s ease}

.log{
  background:#080808;border:1px solid #181818;border-radius:8px;
  padding:11px 14px;height:210px;overflow-y:auto;
  font-family:Consolas,'Courier New',monospace;font-size:11.5px;line-height:1.7;
  color:var(--fdim);white-space:pre-wrap;word-break:break-all;
}
.lg{color:var(--green)} .lr{color:var(--red)} .lb{color:var(--blue);font-weight:700}

/* ── result panel ── */
.rpanel{
  background:var(--surf);border:1px solid #1d3a26;border-radius:var(--r);
  padding:20px 22px;margin-bottom:20px;display:none;
}
.rpanel.show{display:block}
.rhdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.rtitle{font-size:15px;font-weight:700;color:var(--green)}
.btndl{
  background:var(--gdd);color:var(--green);border:1px solid var(--green);
  border-radius:7px;padding:7px 18px;font-size:13px;font-weight:600;
  cursor:pointer;text-decoration:none;transition:background .18s;
}
.btndl:hover{background:#174a2a}
.rimgbox{
  width:100%;overflow-x:auto;background:#090909;
  border-radius:8px;padding:14px;text-align:center;
}
.rimgbox img{max-width:100%;border-radius:6px;box-shadow:0 4px 28px rgba(0,0,0,.7)}
</style>
</head>
<body>

<header>
  <div class="logo">🗺</div>
  <div>
    <div class="htitle">Cadastral Panorama Builder</div>
    <div class="hsub">AI-Based Panoramic Reconstruction — Masters Thesis</div>
  </div>
</header>

<main>

<!-- toolbar -->
<div class="toolbar">
  <div class="selall on" id="selAll" onclick="toggleAll()">
    <div class="chk"></div><span>Select All</span>
  </div>
  <div class="badge" id="badge"></div>
  <div class="spc"></div>
  <button class="btn-gen" id="btnGen" onclick="generate()">GENERATE PANORAMA</button>
</div>

<!-- map grid -->
<div class="grid" id="grid"></div>

<!-- progress panel -->
<div class="ppanel" id="pp">
  <div class="steps">
    <div class="step" id="s1"><div class="sc">1</div><span>Edge Orientation</span></div>
    <div class="sln" id="ln1"></div>
    <div class="step" id="s2"><div class="sc">2</div><span>Panorama Assembly</span></div>
    <div class="sln" id="ln2"></div>
    <div class="step" id="s3"><div class="sc">3</div><span>Done</span></div>
  </div>
  <div class="track"><div class="bar" id="bar"></div></div>
  <div class="log" id="log"></div>
</div>

<!-- result panel -->
<div class="rpanel" id="rp">
  <div class="rhdr">
    <div class="rtitle">✓ &nbsp;Panorama ready</div>
    <a class="btndl" id="dlbtn" href="#" download="panorama.png">⬇ &nbsp;Download PNG</a>
  </div>
  <div class="rimgbox"><img id="rimg" src="" alt="Panorama result"></div>
</div>

</main>

<script>
const MAPS = __MAPS__;
let sel     = new Set(MAPS);
let ws      = null;
let jobDone = false;

/* ── build grid ─────────────────────────────────────────────────────── */
(function buildGrid(){
  const g = document.getElementById('grid');
  MAPS.forEach(m => {
    const d = document.createElement('div');
    d.className = 'card on';
    d.id = 'c' + m;
    d.onclick = () => toggle(m);
    d.innerHTML =
      `<div class="thumb">
         <img src="/thumb/${m}" alt="Map ${m}"
              onerror="this.parentNode.innerHTML='<span class=\\'nop\\'>no preview</span>'">
       </div>
       <div class="cfoot">
         <span class="mlbl">Map ${m}</span>
         <div class="dot"></div>
       </div>`;
    g.appendChild(d);
  });
  refresh();
})();

/* ── selection ──────────────────────────────────────────────────────── */
function toggle(m){
  if(sel.has(m)){ if(sel.size<=2) return; sel.delete(m); }
  else sel.add(m);
  document.getElementById('c'+m).classList.toggle('on', sel.has(m));
  refresh();
}
function toggleAll(){
  sel = (sel.size===MAPS.length) ? new Set(MAPS.slice(0,2)) : new Set(MAPS);
  MAPS.forEach(m => document.getElementById('c'+m).classList.toggle('on', sel.has(m)));
  refresh();
}
function refresh(){
  document.getElementById('badge').textContent = `${sel.size} / ${MAPS.length} maps selected`;
  document.getElementById('selAll').classList.toggle('on', sel.size===MAPS.length);
}

/* ── step indicator ─────────────────────────────────────────────────── */
function setStep(n){
  [1,2,3].forEach(i=>{
    const e=document.getElementById('s'+i);
    e.classList.remove('active','done');
    if(i<n)  e.classList.add('done');
    if(i===n) e.classList.add('active');
  });
  [1,2].forEach(i=>document.getElementById('ln'+i).classList.toggle('done',i<n));
}
function setBar(pct){ document.getElementById('bar').style.width=pct+'%' }

/* ── log ────────────────────────────────────────────────────────────── */
function addLog(text, cls){
  const b=document.getElementById('log');
  const s=document.createElement('span');
  if(cls) s.className=cls;
  s.textContent=text;
  b.appendChild(s);
  b.scrollTop=b.scrollHeight;
}

/* ── generate ───────────────────────────────────────────────────────── */
async function generate(){
  const btn=document.getElementById('btnGen');
  btn.disabled=true;
  document.getElementById('pp').classList.add('show');
  document.getElementById('rp').classList.remove('show');
  document.getElementById('log').innerHTML='';
  setStep(1); setBar(4);

  jobDone = false;
  if(ws){ ws.close(); ws=null; }

  const res = await fetch('/generate',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({maps:[...sel]})
  });
  const {job_id} = await res.json();

  const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${wsProto}//${location.host}/ws/${job_id}`);

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if(msg.type==='log'){
      const t = msg.text;
      let c='';
      if(/saved|✓|winner|done/i.test(t))          c='lg';
      else if(/^[\s=]*(?:pair|===|stage|step)/i.test(t)) c='lb';
      else if(/error/i.test(t))                     c='lr';
      addLog(t, c);
    }
    else if(msg.type==='step'){
      setStep(msg.step);
      setBar(msg.step===1?15 : msg.step===2?70 : 100);
    }
    else if(msg.type==='progress'){
      setBar(Math.round(msg.value*100));
    }
    else if(msg.type==='done'){
      jobDone = true;
      setStep(3); setBar(100);
      btn.disabled=false;
      showResult(msg.job_id);
    }
    else if(msg.type==='error'){
      addLog('\nERROR:\n'+msg.text, 'lr');
      btn.disabled=false;
    }
  };
  ws.onerror = ()=>{ if(!jobDone){ addLog('Connection error.','lr'); btn.disabled=false; } };
  ws.onclose = ()=>{ if(!jobDone){ addLog('Connection closed unexpectedly.','lr'); btn.disabled=false; } };
}

function showResult(job_id){
  const rp=document.getElementById('rp');
  rp.classList.add('show');
  const u=`/result/${job_id}?t=${Date.now()}`;
  document.getElementById('rimg').src=u;
  document.getElementById('dlbtn').href=u;
  rp.scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Cadastral Panorama Builder  —  FastAPI")
    print("  Open in browser:  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
