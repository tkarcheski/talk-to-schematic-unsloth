"""Local read-only dashboard for a training run (issue #3).

Serves http://127.0.0.1:8890 and only reads files: the run's training_manifest.json,
the newest trainer_state.json (run root or checkpoint-*/), and the newest matching
progress log in results/. It never signals or writes to the training process.

    python scripts/training_dashboard.py [--run outputs/<name>] [--port 8890]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)\]")


def newest_run(outputs: Path) -> Path:
    runs = [p for p in outputs.iterdir() if (p / "training_manifest.json").is_file()]
    if not runs:
        raise SystemExit(f"No runs with training_manifest.json under {outputs}")
    return max(runs, key=lambda p: (p / "training_manifest.json").stat().st_mtime)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def progress_log(run: Path, explicit: Path | None = None) -> Path | None:
    """Discover logs on every refresh, including files created after startup."""
    if explicit is not None:
        return explicit
    return max((run.parent.parent / "results").glob(f"{run.name}*.log"),
               key=lambda path: path.stat().st_mtime, default=None)


def trainer_state(run: Path) -> dict | None:
    """Mid-run, trainer_state.json only exists inside checkpoints; take the furthest one."""
    states = [s for s in (read_json(p) for p in [run / "trainer_state.json", *run.glob("checkpoint-*/trainer_state.json")]) if s]
    return max(states, key=lambda s: s.get("global_step", 0), default=None)


def progress(log: Path | None, total_steps: int | None) -> dict | None:
    """Last tqdm bar in the log; when the step count is known, skip data-loading bars."""
    if not log or not log.is_file():
        return None
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 512_000))  # the manifest dump at the end is large
        tail = handle.read().decode(errors="replace")
    matches = [m for m in PROGRESS.findall(tail) if not total_steps or int(m[1]) == total_steps]
    if not matches:
        return None
    step, total, elapsed, remaining, rate, unit = matches[-1]
    return {"step": int(step), "total": int(total), "elapsed": elapsed, "remaining": remaining,
            "rate": f"{rate} {unit}", "log": str(log), "log_mtime": log.stat().st_mtime}


def process_alive(run: Path) -> bool:
    """True when a training process (not this dashboard) has this run's dir on its command line."""
    own = str(os.getpid())
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        if cmdline.parent.name == own:
            continue
        try:
            text = cmdline.read_bytes()
            if run.name.encode() in text and b"train" in text and b"training_dashboard" not in text:
                return True
        except OSError:
            continue
    return False


def gpu() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        name, used, total, util = (part.strip() for part in line.split(","))
        rows.append({"name": name, "mem_used_mb": int(used), "mem_total_mb": int(total), "util_pct": int(util)})
    return rows


def gpu_owners() -> list[str]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        return [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        return ["unavailable"]


def snapshot(run: Path, log: Path | None) -> dict:
    manifest = read_json(run / "training_manifest.json") or {}
    state = trainer_state(run) or {}
    history = state.get("log_history", [])
    train = [e for e in history if "loss" in e]
    eval_entries = [e for e in history if "eval_loss" in e]
    config = manifest.get("config", {})
    total_steps = state.get("max_steps") or (config.get("max_steps") if (config.get("max_steps") or -1) > 0 else None)
    return {
        "run": str(run),
        "status": manifest.get("status", "unknown"),
        "alive": process_alive(run),
        "error_type": manifest.get("error_type"),
        "signal": manifest.get("signal"),
        "global_step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "epochs": config.get("epochs"),
        "progress": progress(log, total_steps),
        "series": {key: [[e["step"], e.get(key)] for e in train if e.get(key) is not None]
                   for key in ("loss", "learning_rate", "grad_norm")},
        "eval_loss": [[e["step"], e["eval_loss"]] for e in eval_entries],
        "gpu": gpu(),
        "gpu_owners": gpu_owners(),
        "config": {key: config.get(key) for key in ("lr", "rank", "gradient_accumulation", "max_seq", "max_image_size")},
        "elapsed_seconds": manifest.get("elapsed_seconds"),
    }


def queue_snapshot(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                "SELECT id,kind,status,elapsed,reason,metrics FROM jobs ORDER BY created DESC LIMIT 100")]
    except sqlite3.Error:
        return [{"id": "", "kind": "queue", "status": "unavailable", "elapsed": 0,
                 "reason": "Cannot read job ledger"}]


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Training Dashboard</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e3df;--accent:#2f6fde;--ok:#1f8a4c;--bad:#c4372c;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--fg:#ecece8;--muted:#9a9a93;--line:#2e2e2b;--accent:#6b9cf0;--ok:#4cc27e;--bad:#ef6a5f;--warn:#e0a84a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:18px;margin:0 0 4px} .muted{color:var(--muted)} code{font-size:12px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.k{font-size:12px;color:var(--muted)} .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.charts{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
svg{width:100%;height:160px;display:block} .bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px}
.bar>div{height:100%;background:var(--accent);border-radius:3px}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}
</style>
<main><h1>Training Dashboard</h1><div class=muted id=run></div>
<div class=grid id=tiles></div><div class=charts id=charts></div>
<h2>Research and training queue</h2><div id=queue class=card></div><p class=muted id=foot></p></main>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??"—").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function chart(title,pts,fmt){
  if(!pts.length) return `<div class=card><div class=k>${title}</div><p class=muted>No data yet</p></div>`;
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]), x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  const W=320,H=140,px=x=>(x1===x0?0:(x-x0)/(x1-x0))*W, py=y=>H-(y1===y0?.5:(y-y0)/(y1-y0))*H;
  const d=pts.map((p,i)=>(i?"L":"M")+px(p[0]).toFixed(1)+" "+py(p[1]).toFixed(1)).join("");
  return `<div class=card><div class=k>${title} · last ${fmt(ys.at(-1))} · min ${fmt(y0)} · max ${fmt(y1)}</div>
  <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio=none><path d="${d}" fill=none stroke="var(--accent)" stroke-width=1.5 vector-effect=non-scaling-stroke /></svg>
  <div class=k>step ${x0} → ${x1}</div></div>`;
}
const tile=(k,v,extra="")=>`<div class=card><div class=k>${k}</div><div class=v>${v}</div>${extra}</div>`;
function jobMetrics(raw){
  let metrics; try{metrics=typeof raw==="string"?JSON.parse(raw):raw;}catch{return "Metrics unavailable";}
  if(!metrics||typeof metrics!=="object")return "Metrics unavailable";
  const labels={checkpoint_step:"Checkpoint step",source_count:"Sources",proposal_count:"Proposals",error_code:"Error"};
  return Object.entries(labels).filter(([key])=>metrics[key]!=null).map(([key,label])=>label+": "+String(metrics[key])).join(" · ")||"No checkpoint or source metrics reported";
}
async function tick(){
  let s; try{ s=await (await fetch("/api/state")).json(); }catch(e){ $("foot").textContent="Can't reach the dashboard server."; return; }
  const color={completed:"var(--ok)",failed:"var(--bad)",interrupted:"var(--warn)"}[s.status]||"var(--accent)";
  const status=`<span class=pill style="background:${color}">${esc(s.status)}</span> ${s.alive?"process running":"no process"}`
    +(s.error_type?`<div class=k>${esc(s.error_type)} ${esc(s.signal||"")}</div>`:"");
  const p=s.progress, step=p?.step??s.global_step, total=p?.total??s.max_steps, pct=step&&total?100*step/total:0;
  $("run").innerHTML=`<code>${esc(s.run)}</code>`;
  $("tiles").innerHTML=[
    tile("Status",status),
    tile("Step",`${esc(step)} / ${esc(total)}`,`<div class=bar><div style="width:${pct}%"></div></div>`),
    tile("Epoch",`${s.epoch!=null?s.epoch.toFixed(2):"—"} / ${esc(s.epochs)}`),
    tile("Speed",esc(p?.rate)),
    tile("GPU process owners",esc((s.gpu_owners||[]).join(", ")||"none")),
    tile("Time left",esc(p?.remaining)),
    ...s.gpu.map(g=>tile(esc(g.name),`${(g.mem_used_mb/1024).toFixed(1)} / ${(g.mem_total_mb/1024).toFixed(0)} GB`,
      `<div class=k>${g.util_pct}% util</div><div class=bar><div style="width:${100*g.mem_used_mb/g.mem_total_mb}%"></div></div>`)),
  ].join("");
  const f=v=>v==null?"—":Math.abs(v)<1e-3&&v!==0?v.toExponential(2):v.toFixed(4);
  $("charts").innerHTML=chart("Loss",s.series.loss,f)+chart("Learning rate",s.series.learning_rate,f)
    +chart("Gradient norm",s.series.grad_norm,f)+chart("Validation loss",s.eval_loss,f);
  $("queue").innerHTML=(s.jobs||[]).map(j=>`<p><strong>${esc(j.kind)}</strong> · ${esc(j.status)} · ${Number(j.elapsed).toFixed(1)} s <span class=muted>${esc(j.reason)}</span><br><small>${esc(jobMetrics(j.metrics))}</small></p>`).join("")||"No queued jobs. Submit jobs with scripts/research_scheduler.py.";
  $("foot").textContent=`Updated ${new Date().toLocaleTimeString()} · refreshes every 10 s · read-only`;
}
tick(); setInterval(tick,10000);
</script>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory (default: newest under outputs/)")
    parser.add_argument("--log", type=Path, help="progress log (default: newest results/<run name>*.log)")
    parser.add_argument("--follow", action="store_true", help="follow newest manifest under the selected run parent")
    parser.add_argument("--ledger", type=Path, help="read-only research scheduler SQLite ledger")
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()
    run = (args.run or newest_run(ROOT / "outputs")).resolve()
    log = progress_log(run, args.log)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/state":
                selected = newest_run(run.parent) if args.follow else run
                selected_log = progress_log(selected, args.log)
                body, kind = json.dumps({**snapshot(selected, selected_log), "jobs": queue_snapshot(args.ledger)}).encode(), "application/json"
            elif self.path == "/":
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    print(f"Dashboard for {run}\nlog: {log}\nhttp://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
