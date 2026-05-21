"""Flask routes for the GMS Mission Control web app."""

from __future__ import annotations

import csv
import io
import json
import time
import traceback
from pathlib import Path

from flask import Response, jsonify, render_template, request
from markupsafe import Markup


PAGE_ROUTES = {
    "/": ("dashboard.html", "db"),
    "/optimize": ("optimize.html", "opt"),
    "/map": ("map.html", "map"),
    "/analysis": ("analysis.html", "an"),
    "/alerts": ("alerts.html", "al"),
    "/about": ("about.html", "ab"),
}


def _nav_icon(kind: str) -> str:
    icons = {
        "db": '<rect x="1" y="1" width="5" height="5" rx="1" fill="currentColor" opacity=".7"/><rect x="8" y="1" width="5" height="5" rx="1" fill="currentColor"/><rect x="1" y="8" width="5" height="5" rx="1" fill="currentColor" opacity=".4"/><rect x="8" y="8" width="5" height="5" rx="1" fill="currentColor" opacity=".4"/>',
        "opt": '<path d="M3 3 L13 3 L11 9 L5 9 Z" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="8" cy="8" r="1.5" fill="currentColor"/>',
        "map": '<circle cx="3" cy="10" r="2" fill="currentColor" opacity=".5"/><circle cx="7" cy="6" r="2" fill="currentColor" opacity=".8"/><circle cx="11" cy="3" r="2" fill="currentColor"/><line x1="3" y1="10" x2="7" y2="6" stroke="currentColor" stroke-width="1.2"/><line x1="7" y1="6" x2="11" y2="3" stroke="currentColor" stroke-width="1.2"/>',
        "an": '<polyline points="1,11 4,6 7,8 10,3 13,5" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        "al": '<rect x="1" y="2" width="12" height="10" rx="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="4" y1="6" x2="10" y2="6" stroke="currentColor" stroke-width="1"/><line x1="4" y1="8" x2="8" y2="8" stroke="currentColor" stroke-width="1"/>',
        "ab": '<circle cx="7" cy="7" r="5.5" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="7" y1="6" x2="7" y2="10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="7" cy="4" r=".7" fill="currentColor"/>',
    }
    return icons[kind]


def _navbar(engine, active: str) -> str:
    t = engine.t
    time_label = engine.time_label(t)
    high_count = int((engine.label[:, t] == 2).sum())
    status_class = "p-hi" if high_count else "p-ok"
    status_text = f"! {high_count} HIGH" if high_count else "ALL STABLE"
    noise_class = "p-ns" if engine.noise_on else "p-ok"
    noise_text = "NOISE ON" if engine.noise_on else "CLEAN DATA"
    danger_count = len([
        a for a in engine.alert_history
        if a.get("level") == "danger" and a.get("status") != "resolved"
    ])

    pages = [
        ("/", "db", "Dashboard"),
        ("/optimize", "opt", "Optimize"),
        ("/map", "map", "Sensor Map"),
        ("/analysis", "an", "Analysis"),
        ("/alerts", "al", "Alerts"),
        ("/about", "ab", "About"),
    ]

    links = []
    for href, key, label in pages:
        badge = ""
        if key == "al" and danger_count:
            badge = f'<span class="nbdg">{danger_count}</span>'
        active_class = " active" if key == active else ""
        links.append(
            f'<a href="{href}" class="nl{active_class}">'
            f'<svg viewBox="0 0 14 14" fill="none">{_nav_icon(key)}</svg>'
            f"{label}{badge}</a>"
        )

    return f"""<nav>
  <a href="/" class="brand">
    <div class="bico"><svg viewBox="0 0 16 16" fill="none">
      <circle cx="4" cy="12" r="2.2" fill="#388BFD"/>
      <circle cx="8" cy="7" r="2.2" fill="#D29922"/>
      <circle cx="12" cy="3" r="2.2" fill="#F85149"/>
      <line x1="4" y1="12" x2="8" y2="7" stroke="#388BFD" stroke-width="1.2"/>
      <line x1="8" y1="7" x2="12" y2="3" stroke="#D29922" stroke-width="1.2"/>
    </svg></div>
    <div><div class="bnm">GMS</div><div class="bsb">Mission Control | N={engine.N}</div></div>
  </a>
  <div style="display:flex;align-items:stretch;height:100%;flex:1">{''.join(links)}</div>
  <div class="nav-r">
    <span class="pill p-live">LIVE</span>
    <span class="pill p-t" id="nav-t">{time_label}</span>
    <span class="pill {noise_class}" id="nav-noise">{noise_text}</span>
    <span class="pill {status_class}" id="nav-st">{status_text}</span>
  </div>
</nav>"""


def _render_page(engine, template_name: str, active: str):
    return render_template(template_name, NAV=Markup(_navbar(engine, active)))


def _json_body() -> dict:
    return request.get_json(silent=True) or {}


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf(lines):
    """Build a small text-only PDF without external dependencies."""
    y = 780
    content = ["BT", "/F1 10 Tf", "50 800 Td"]
    for line in lines:
        content.append(f"0 -14 Td ({_pdf_escape(line)[:110]}) Tj")
        y -= 14
        if y < 50:
            break
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode("ascii"))
        out.write(obj)
        out.write(b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.write(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    return out.getvalue()


def register_routes(app, engine) -> None:
    """Register all HTML, JSON API, export, and SSE routes."""

    for route, (template_name, active) in PAGE_ROUTES.items():
        app.add_url_rule(
            route,
            endpoint=f"page_{active}",
            view_func=lambda template_name=template_name, active=active: _render_page(
                engine, template_name, active
            ),
        )

    @app.get("/api/state")
    def api_state():
        return jsonify(engine.frame_data())

    @app.get("/api/alerts")
    def api_alerts():
        return jsonify(engine.alert_history)

    @app.get("/api/alerts/analytics")
    def api_alert_analytics():
        return jsonify(engine.alert_analytics())

    @app.get("/api/alerts/export")
    def api_alert_export():
        fmt = request.args.get("format", "csv").lower()
        rows = []
        for alert in engine.alert_history:
            rows.append({
                "id": alert.get("id", ""),
                "t": alert.get("t", ""),
                "time": alert.get("time_label", ""),
                "level": alert.get("level", ""),
                "status": alert.get("status", ""),
                "category": alert.get("category", ""),
                "nodes": ",".join(f"N{node}" for node in alert.get("nodes", [])),
                "message": alert.get("msg", ""),
                "action": alert.get("action", ""),
            })

        if fmt == "pdf":
            lines = ["GMS Alert Export", ""]
            for row in rows:
                lines.append(
                    f"{row['id']} {row['time'] or row['t']} {row['level'].upper()} {row['status']} "
                    f"{row['nodes']} - {row['message']}"
                )
            return Response(
                _simple_pdf(lines),
                mimetype="application/pdf",
                headers={"Content-Disposition": "attachment; filename=gms_alerts.pdf"},
            )

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["id", "t", "time", "level", "status", "category", "nodes", "message", "action"])
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=gms_alerts.csv"},
        )

    @app.post("/api/alerts/<alert_id>/status")
    def api_alert_status(alert_id):
        status = str(_json_body().get("status", "")).lower()
        alert = engine.set_alert_status(alert_id, status)
        if alert is None:
            return jsonify(ok=False, error="alert not found or invalid status"), 404
        return jsonify(ok=True, alert=alert)

    @app.post("/api/alerts/clear")
    def api_clear_alerts():
        engine.clear_alerts()
        return jsonify(ok=True)

    @app.post("/api/alerts/hysteresis")
    def api_alert_hysteresis():
        body = _json_body()
        return jsonify(ok=True, policy=engine.set_hysteresis(body.get("margin", engine.hysteresis_margin)))

    @app.post("/api/alerts/ai_tune")
    def api_alert_ai_tune():
        body = _json_body()
        result = engine.tune_alert_policy(
            iterations=int(body.get("iterations", 350)),
            seed=int(body.get("seed", 42)),
            target_far=float(body.get("target_far", 0.02)),
            min_recall=float(body.get("min_recall", 0.25)),
            apply=bool(body.get("apply", False)),
        )
        return jsonify(success=True, **result)

    @app.post("/api/play")
    def api_play():
        engine.play()
        return jsonify(ok=True)

    @app.post("/api/pause")
    def api_pause():
        engine.pause()
        return jsonify(ok=True)

    @app.post("/api/reset")
    def api_reset():
        engine.reset()
        return jsonify(ok=True)

    @app.post("/api/jump")
    def api_jump():
        engine.jump(int(_json_body().get("t", 0)))
        return jsonify(ok=True)

    @app.post("/api/step")
    def api_step():
        engine.step(int(_json_body().get("dir", 1)))
        return jsonify(ok=True)

    @app.post("/api/speed")
    def api_speed():
        engine.speed = float(_json_body().get("speed", 0.25))
        return jsonify(ok=True)

    @app.post("/api/params")
    def api_params():
        engine.rerun(_json_body())
        return jsonify(ok=True)

    @app.post("/api/trigger_event")
    def api_trigger_event():
        engine.trigger(int(_json_body().get("idx", 0)))
        return jsonify(ok=True)

    @app.post("/api/toggle_noise")
    def api_toggle_noise():
        engine.toggle_noise(bool(_json_body().get("on", False)))
        return jsonify(ok=True)

    @app.post("/api/optimize")
    def api_optimize():
        body = _json_body()
        try:
            from core.optimizer import GMSOptimizer
            from data.loader import SimulatedData

            optimizer = GMSOptimizer(SimulatedData(), verbose=False)
            optimizer.target_recall = float(body.get("target_recall", 0.10))
            optimizer.target_fp_rate = float(body.get("target_fp_rate", 0.02))
            optimizer.optimize(
                n_iter=int(body.get("iterations", 50)),
                seed=int(body.get("seed", 42)),
            )

            _, best_metrics = optimizer._evaluate_params(optimizer.best_params)
            raw_history = optimizer.optimization_log
            stride = max(1, len(raw_history) // 160)
            compact_history = []
            best_seen = None
            for idx, item in enumerate(raw_history):
                if idx % stride and idx != len(raw_history) - 1:
                    continue
                loss = float(item.get("loss", 0))
                best_seen = loss if best_seen is None else min(best_seen, loss)
                metrics = item.get("metrics", {})
                compact_history.append(
                    {
                        "eval": idx + 1,
                        "loss": loss,
                        "best_loss": float(best_seen),
                        "accuracy": float(metrics.get("accuracy", 0)),
                        "precision": float(metrics.get("precision", 0)),
                        "recall": float(metrics.get("recall", 0)),
                        "far": float(metrics.get("far", 0)),
                        "f1": float(metrics.get("f1", 0)),
                    }
                )

            return jsonify(
                {
                    "success": True,
                    "best_params": optimizer.best_params,
                    "best_loss": float(optimizer.best_loss),
                    "metrics": {
                        "accuracy": float(best_metrics.get("accuracy", 0)),
                        "precision": float(best_metrics.get("precision", 0)),
                        "recall": float(best_metrics.get("recall", 0)),
                        "far": float(best_metrics.get("far", 0)),
                        "f1": float(best_metrics.get("f1", 0)),
                        "tp": int(best_metrics.get("tp", 0)),
                        "fp": int(best_metrics.get("fp", 0)),
                        "fn": int(best_metrics.get("fn", 0)),
                        "tn": int(best_metrics.get("tn", 0)),
                    },
                    "history": compact_history,
                    "iterations_evaluated": len(optimizer.optimization_log),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            Path("outputs").mkdir(exist_ok=True)
            Path("outputs/optimize_debug.log").write_text(tb, encoding="utf-8")
            return jsonify({"success": False, "error": str(exc), "traceback": tb}), 500

    @app.post("/api/apply_optimized_params")
    def api_apply_optimized_params():
        params = _json_body().get("params", {})
        applied = engine.rerun(params)
        return jsonify(
            success=True,
            message="Parameters applied",
            applied=applied,
            alert_policy=engine.frame_data()["alert_policy"],
        )

    @app.get("/export")
    def export_data():
        try:
            import pandas as pd

            Path("outputs").mkdir(exist_ok=True)
            filename = "gms_noise.csv" if engine.noise_on else "gms_clean.csv"
            file_path = Path("outputs") / filename
            pd.DataFrame(engine.logs).to_csv(file_path, index=False)
            return jsonify(status="saved", file=str(file_path))
        except Exception as exc:
            return jsonify(status="error", error=str(exc)), 500

    @app.post("/reset_logs")
    def reset_logs():
        engine.logs = []
        engine.last_logged_t = None
        return jsonify(status="logs cleared")

    @app.get("/stream")
    def stream():
        q = engine.subscribe()
        q.append(json.dumps({"type": "frame", "data": engine.frame_data()}))

        def gen():
            try:
                while True:
                    if q:
                        yield f"data: {q.pop(0)}\n\n"
                    else:
                        yield ": hb\n\n"
                        time.sleep(0.04)
            except GeneratorExit:
                engine.unsubscribe(q)

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
