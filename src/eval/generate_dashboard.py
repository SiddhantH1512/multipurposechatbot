# src/eval/generate_dashboard.py
"""
Generates a beautiful, self-contained HTML evaluation dashboard from
a Self-RAG RAGAS report JSON.

Can be called from eval_self_rag.py after evaluation, or run standalone:
    python -m src.eval.generate_dashboard eval_results/self_rag_report_<ts>.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from datetime import datetime


# ── Colour helpers ───────────────────────────────────────────────────────────

def _score_color(v: float | None) -> str:
    if v is None:
        return "#64748b"
    if v >= 0.75:
        return "#10b981"
    if v >= 0.50:
        return "#f59e0b"
    return "#ef4444"

def _score_bg(v: float | None) -> str:
    if v is None:
        return "rgba(100,116,139,0.12)"
    if v >= 0.75:
        return "rgba(16,185,129,0.12)"
    if v >= 0.50:
        return "rgba(245,158,11,0.12)"
    return "rgba(239,68,68,0.12)"

def _grade_badge(grade: str) -> str:
    colors = {
        "fully_supported":    ("#10b981", "Fully Supported"),
        "partially_supported":("#f59e0b", "Partial"),
        "not_supported":      ("#ef4444", "Not Supported"),
        "error":              ("#ef4444", "Error"),
        "unknown":            ("#64748b", "N/A"),
    }
    color, label = colors.get(grade, ("#64748b", grade))
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}44;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;letter-spacing:0.04em">{label}</span>'

def _diff_badge(diff: str) -> str:
    colors = {"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"}
    c = colors.get(diff.lower(), "#64748b")
    return f'<span style="background:{c}22;color:{c};border:1px solid {c}44;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600">{diff.title()}</span>'

def _pct(v: float | None) -> str:
    return f"{v*100:.1f}%" if v is not None else "—"

def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"

def _bar_svg(value: float | None, width: int = 120) -> str:
    if value is None:
        return '<span style="color:#64748b;font-size:12px">—</span>'
    pct    = max(0, min(1, value))
    filled = int(pct * width)
    color  = _score_color(value)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:{width}px;height:6px;background:#1e293b;border-radius:3px;overflow:hidden">'
        f'<div style="width:{filled}px;height:6px;background:{color};border-radius:3px;'
        f'box-shadow:0 0 6px {color}88"></div></div>'
        f'<span style="font-size:12px;color:{color};font-weight:600;font-family:\'JetBrains Mono\',monospace">'
        f'{value:.3f}</span></div>'
    )

def _sparkline(values: list[float | None], width=80, height=24) -> str:
    """Tiny inline SVG sparkline."""
    valid = [v for v in values if v is not None]
    if not valid:
        return ""
    mn, mx = min(valid), max(valid)
    span = mx - mn or 0.001
    pts = []
    step = width / max(len(values) - 1, 1)
    for i, v in enumerate(values):
        if v is None:
            continue
        x = i * step
        y = height - ((v - mn) / span) * (height - 2) - 1
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="overflow:visible">'
        f'<polyline points="{poly}" fill="none" stroke="#6366f1" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ── Main dashboard generator ─────────────────────────────────────────────────

def generate_dashboard(report: dict, output_dir: Path, timestamp: str) -> str:
    items       = report.get("per_item", [])
    avg         = report.get("avg_metrics", {})
    total       = report.get("total_questions", len(items))
    pipeline    = report.get("pipeline", "Self-RAG")
    ts_fmt      = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%d %b %Y, %H:%M")

    # ── Aggregate Self-RAG stats ─────────────────────────────────────────────
    total_retries   = sum(i.get("retry_count", 0)   for i in items)
    total_rewrites  = sum(i.get("rewrite_count", 0) for i in items)
    errors          = sum(1 for i in items if i.get("error"))
    avg_latency     = (sum(i.get("latency", 0) for i in items) / max(len(items), 1))
    faith_counts    = {"fully_supported": 0, "partially_supported": 0, "not_supported": 0, "error": 0}
    for i in items:
        g = i.get("faithfulness_grade", "unknown")
        faith_counts[g if g in faith_counts else "error"] += 1

    diff_groups = {"easy": [], "medium": [], "hard": []}
    for i in items:
        d = i.get("difficulty", "easy").lower()
        if d in diff_groups:
            score = i.get("ragas_answer_correctness")
            if score is not None:
                diff_groups[d].append(score)

    def avg_list(lst): return sum(lst)/len(lst) if lst else None

    diff_avgs = {d: avg_list(v) for d, v in diff_groups.items()}

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]
    metric_labels = {
        "faithfulness":        "Faithfulness",
        "answer_relevancy":    "Answer Relevancy",
        "context_precision":   "Context Precision",
        "context_recall":      "Context Recall",
        "answer_correctness":  "Answer Correctness",
    }

    # ── KPI cards ────────────────────────────────────────────────────────────
    def kpi_card(label, value, sub="", color="#6366f1"):
        return f"""
        <div class="kpi-card" style="border-top:3px solid {color}">
          <div class="kpi-value" style="color:{color}">{value}</div>
          <div class="kpi-label">{label}</div>
          {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
        </div>"""

    kpi_html = "".join([
        kpi_card("Total Questions", total, pipeline, "#6366f1"),
        kpi_card("Avg Correctness", _pct(avg.get("answer_correctness")),
                 "ground-truth aligned", _score_color(avg.get("answer_correctness"))),
        kpi_card("Avg Faithfulness", _pct(avg.get("faithfulness")),
                 "doc-grounded answers", _score_color(avg.get("faithfulness"))),
        kpi_card("Avg Latency", f"{avg_latency:.1f}s", "per question", "#0ea5e9"),
        kpi_card("Self-RAG Retries", total_retries, "faithfulness re-gens", "#f59e0b"),
        kpi_card("Query Rewrites", total_rewrites, "usefulness re-tries", "#8b5cf6"),
        kpi_card("Errors", errors, "pipeline failures", "#ef4444" if errors else "#10b981"),
    ])

    # ── Metric gauge cards ───────────────────────────────────────────────────
    def gauge_card(key):
        v   = avg.get(key)
        pct = (v or 0) * 100
        c   = _score_color(v)
        # SVG arc gauge
        r, cx, cy = 40, 55, 55
        circ = 2 * math.pi * r
        dash = circ * (pct / 100)
        gap  = circ - dash
        return f"""
        <div class="gauge-card">
          <svg width="110" height="75" viewBox="0 0 110 80">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e293b" stroke-width="8"
              stroke-dasharray="{circ*0.75:.1f} {circ*0.25:.1f}"
              stroke-dashoffset="{circ*0.375:.1f}" stroke-linecap="round"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{c}" stroke-width="8"
              stroke-dasharray="{dash*0.75:.1f} {circ - dash*0.75:.1f}"
              stroke-dashoffset="{circ*0.375:.1f}" stroke-linecap="round"
              style="filter:drop-shadow(0 0 4px {c}88)"/>
            <text x="{cx}" y="{cy+6}" text-anchor="middle" fill="{c}"
              font-size="14" font-weight="700" font-family="JetBrains Mono,monospace">
              {_pct(v)}</text>
          </svg>
          <div class="gauge-label">{metric_labels.get(key, key)}</div>
        </div>"""

    gauges_html = "".join(gauge_card(k) for k in metric_keys)

    # ── Faithfulness pie (donut via SVG) ─────────────────────────────────────
    faith_colors = {"fully_supported": "#10b981", "partially_supported": "#f59e0b",
                    "not_supported": "#ef4444", "error": "#64748b"}
    faith_labels = {"fully_supported": "Fully Supported", "partially_supported": "Partially Supported",
                    "not_supported": "Not Supported", "error": "Error"}
    total_faith = sum(faith_counts.values()) or 1

    def donut_slice(pct_start, pct, color, r=40, cx=55, cy=55):
        if pct <= 0:
            return ""
        circ = 2 * math.pi * r
        dash = circ * pct
        offset = circ - circ * pct_start
        return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
                f'stroke-width="16" stroke-dasharray="{dash:.2f} {circ-dash:.2f}" '
                f'stroke-dashoffset="{offset:.2f}" stroke-linecap="butt"/>')

    donut_svg = '<svg width="110" height="110" viewBox="0 0 110 110">'
    cursor = 0
    for key, cnt in faith_counts.items():
        pct = cnt / total_faith
        donut_svg += donut_slice(cursor, pct, faith_colors[key])
        cursor += pct
    donut_svg += '<circle cx="55" cy="55" r="30" fill="#0f172a"/>'
    donut_svg += f'<text x="55" y="50" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="700" font-family="JetBrains Mono,monospace">{faith_counts["fully_supported"]}</text>'
    donut_svg += f'<text x="55" y="65" text-anchor="middle" fill="#64748b" font-size="9">fully ok</text>'
    donut_svg += '</svg>'

    faith_legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
        f'<div style="width:10px;height:10px;border-radius:2px;background:{faith_colors[k]}"></div>'
        f'<span style="color:#94a3b8;font-size:12px">{faith_labels[k]}</span>'
        f'<span style="color:{faith_colors[k]};font-weight:700;font-size:12px;margin-left:auto">{faith_counts[k]}</span>'
        f'</div>'
        for k in faith_counts
    )

    # ── Difficulty breakdown bars ────────────────────────────────────────────
    diff_html = ""
    for d, col in [("easy", "#10b981"), ("medium", "#f59e0b"), ("hard", "#ef4444")]:
        v = diff_avgs.get(d)
        n = len(diff_groups[d])
        pct_w = int((v or 0) * 180)
        diff_html += f"""
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="color:#e2e8f0;font-size:13px;text-transform:capitalize;font-weight:600">{d}</span>
            <span style="color:{col};font-size:12px;font-family:JetBrains Mono,monospace">
              {_pct(v)} <span style="color:#475569">({n} Qs)</span></span>
          </div>
          <div style="background:#1e293b;height:6px;border-radius:3px;overflow:hidden">
            <div style="width:{pct_w}px;max-width:180px;height:6px;background:{col};
              border-radius:3px;box-shadow:0 0 6px {col}88"></div>
          </div>
        </div>"""

    # ── Per-question table rows ──────────────────────────────────────────────
    table_rows = ""
    for i, item in enumerate(items):
        cols_html = ""
        for mk in metric_keys:
            v = item.get(f"ragas_{mk}")
            cols_html += f'<td>{_bar_svg(v, 80)}</td>'

        faith = item.get("faithfulness_grade", "unknown")
        retry = item.get("retry_count", 0)
        rew   = item.get("rewrite_count", 0)
        lat   = item.get("latency", 0)
        err   = item.get("error")

        retry_badge = (f'<span style="background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44;'
                       f'border-radius:4px;padding:1px 6px;font-size:11px">{retry}×</span>' if retry else
                       '<span style="color:#475569;font-size:11px">—</span>')
        rew_badge   = (f'<span style="background:#8b5cf622;color:#8b5cf6;border:1px solid #8b5cf644;'
                       f'border-radius:4px;padding:1px 6px;font-size:11px">{rew}×</span>' if rew else
                       '<span style="color:#475569;font-size:11px">—</span>')

        row_class = "table-row-error" if err else ("table-row-even" if i % 2 == 0 else "")
        table_rows += f"""
        <tr class="{row_class}" onclick="toggleDetail({i})" style="cursor:pointer">
          <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#6366f1">{item['id']}</td>
          <td>{_diff_badge(item.get('difficulty','easy'))}</td>
          <td style="font-size:11px;color:#94a3b8;max-width:240px">{item.get('question_type','')}</td>
          <td style="font-size:12px;color:#cbd5e1;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{item['question']}">{item['question'][:70]}{"…" if len(item['question'])>70 else ""}</td>
          {cols_html}
          <td>{_grade_badge(faith)}</td>
          <td>{retry_badge}</td>
          <td>{rew_badge}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#0ea5e9">{lat:.1f}s</td>
        </tr>
        <tr id="detail-{i}" style="display:none">
          <td colspan="13" style="padding:0">
            <div class="detail-panel">
              <div class="detail-grid">
                <div>
                  <div class="detail-label">QUESTION</div>
                  <div class="detail-text">{item['question']}</div>
                </div>
                <div>
                  <div class="detail-label">GROUND TRUTH</div>
                  <div class="detail-text" style="color:#10b981">{item.get('ground_truth','—')}</div>
                </div>
                <div style="grid-column:1/-1">
                  <div class="detail-label">PIPELINE ANSWER</div>
                  <div class="detail-text answer-text">{item.get('answer','—')[:600]}{"…" if len(str(item.get('answer','')))>600 else ""}</div>
                </div>
                {"<div style='grid-column:1/-1'><div class='detail-label' style='color:#ef4444'>ERROR</div><div class='detail-text' style='color:#ef4444'>" + str(item.get('error','')) + "</div></div>" if err else ""}
                {"<div style='grid-column:1/-1'><div class='detail-label' style='color:#f59e0b'>CONFLICT NOTE</div><div class='detail-text' style='color:#f59e0b'>" + item.get('conflict_note','') + "</div></div>" if item.get('conflict_note') else ""}
                {"<div style='grid-column:1/-1'><div class='detail-label'>FOLLOW-UP SUGGESTIONS</div><div class='detail-text'>" + "<br>".join(f"• {q}" for q in item.get('follow_ups',[])) + "</div></div>" if item.get('follow_ups') else ""}
              </div>
            </div>
          </td>
        </tr>"""

    # ── Score trend sparklines (per metric across all questions) ─────────────
    sparklines_html = ""
    for mk in metric_keys:
        vals = [item.get(f"ragas_{mk}") for item in items]
        c    = _score_color(avg.get(mk))
        sparklines_html += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
          <div style="width:160px;font-size:12px;color:#94a3b8">{metric_labels[mk]}</div>
          {_sparkline(vals, 180, 20)}
          <span style="color:{c};font-weight:700;font-size:12px;font-family:'JetBrains Mono',monospace;min-width:40px">{_pct(avg.get(mk))}</span>
        </div>"""

    # ── HTML assembly ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PolicyIQ — Self-RAG Evaluation Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #070d1b;
    --surface:   #0f172a;
    --surface2:  #1e293b;
    --border:    #1e293b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --accent:    #6366f1;
    --accent2:   #8b5cf6;
    --green:     #10b981;
    --amber:     #f59e0b;
    --red:       #ef4444;
    --blue:      #0ea5e9;
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Noise texture overlay ── */
  body::before {{
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    background-size: 256px;
    opacity: 0.4;
  }}

  .page-wrap {{ position: relative; z-index: 1; max-width: 1440px; margin: 0 auto; padding: 32px 24px 64px; }}

  /* ── Header ── */
  .header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 40px; padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap; gap: 16px;
  }}
  .header-logo {{
    font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: 6px;
  }}
  .header-title {{
    font-size: 28px; font-weight: 700; color: var(--text);
    letter-spacing: -0.02em; line-height: 1.2;
  }}
  .header-title span {{ color: var(--accent); }}
  .header-meta {{
    text-align: right; color: var(--muted); font-size: 12px; line-height: 1.8;
  }}
  .header-meta strong {{ color: var(--text); }}

  /* ── Section headings ── */
  .section-heading {{
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }}
  .section-heading::after {{
    content: ''; flex: 1; height: 1px; background: var(--border);
  }}

  /* ── KPI grid ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 40px;
  }}
  .kpi-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 18px;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px #00000044; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; letter-spacing: -0.03em; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }}
  .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }}
  .kpi-sub   {{ font-size: 11px; color: #475569; margin-top: 3px; }}

  /* ── Two-column analytics row ── */
  .analytics-row {{
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 20px; margin-bottom: 40px; align-items: start;
  }}

  /* ── Gauge grid ── */
  .gauge-grid {{
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: space-around;
  }}
  .gauge-card {{
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }}
  .gauge-label {{
    font-size: 11px; color: var(--muted); text-align: center;
    max-width: 90px; line-height: 1.3;
  }}

  /* ── Panel card ── */
  .panel {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 24px; height: 100%;
  }}
  .panel-title {{
    font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 16px;
    padding-bottom: 10px; border-bottom: 1px solid var(--border);
  }}

  /* ── Table ── */
  .table-wrap {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; margin-bottom: 40px;
  }}
  .table-header {{
    padding: 18px 24px 14px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }}
  .table-title {{ font-size: 13px; font-weight: 600; color: var(--text); }}
  .table-subtitle {{ font-size: 12px; color: var(--muted); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead th {{
    background: #0d1829; color: var(--muted);
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 10px 12px; text-align: left; white-space: nowrap;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }}
  tbody tr {{ border-bottom: 1px solid #111827; transition: background 0.1s; }}
  tbody tr:hover {{ background: #0d1829 !important; }}
  tbody td {{ padding: 10px 12px; vertical-align: middle; }}
  .table-row-even {{ background: #080f1e; }}
  .table-row-error {{ background: rgba(239,68,68,0.04); }}

  /* ── Expandable detail panel ── */
  .detail-panel {{
    background: #060c1a; border-top: 1px solid var(--border);
    padding: 20px 24px; animation: fadeIn 0.15s ease;
  }}
  .detail-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  .detail-label {{
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px; font-weight: 600;
  }}
  .detail-text {{
    font-size: 13px; color: #cbd5e1; line-height: 1.6;
    font-family: 'Space Grotesk', sans-serif;
  }}
  .answer-text {{
    background: #0d1829; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px;
    font-size: 12px; color: #94a3b8;
  }}

  @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(-4px) }} to {{ opacity:1; transform:none }} }}

  /* ── Search / filter bar ── */
  .filter-bar {{
    display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 24px;
    border-bottom: 1px solid var(--border); background: #080f1e;
  }}
  .filter-input {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: 7px 12px; font-size: 13px; outline: none;
    font-family: 'Space Grotesk', sans-serif; min-width: 200px;
    transition: border-color 0.15s;
  }}
  .filter-input:focus {{ border-color: var(--accent); }}
  .filter-select {{
    background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: 7px 12px; font-size: 13px; outline: none;
    font-family: 'Space Grotesk', sans-serif; cursor: pointer;
  }}
  .filter-label {{ color: var(--muted); font-size: 12px; align-self: center; }}

  /* ── Footer ── */
  .footer {{
    border-top: 1px solid var(--border); padding-top: 20px; margin-top: 20px;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    color: var(--muted); font-size: 12px;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--surface); }}
  ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #475569; }}

  @media (max-width: 900px) {{
    .analytics-row {{ grid-template-columns: 1fr 1fr; }}
    .detail-grid {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 600px) {{
    .analytics-row {{ grid-template-columns: 1fr; }}
    .header {{ flex-direction: column; }}
    .header-meta {{ text-align: left; }}
  }}
</style>
</head>
<body>
<div class="page-wrap">

  <!-- ── Header ── -->
  <header class="header">
    <div>
      <div class="header-logo">PolicyIQ · RAG Evaluation Suite</div>
      <h1 class="header-title">Self-RAG <span>Evaluation</span> Dashboard</h1>
      <div style="color:var(--muted);font-size:13px;margin-top:6px">
        RAGAS metrics · Faithfulness grading · Per-question diagnostics
      </div>
    </div>
    <div class="header-meta">
      <div><strong>Pipeline</strong>&nbsp;&nbsp;{pipeline}</div>
      <div><strong>Run Date</strong>&nbsp;&nbsp;{ts_fmt}</div>
      <div><strong>Questions</strong>&nbsp;&nbsp;{total}</div>
      <div><strong>Embeddings</strong>&nbsp;&nbsp;BAAI/bge-small-en-v1.5</div>
    </div>
  </header>

  <!-- ── KPI row ── -->
  <div class="section-heading">Overview</div>
  <div class="kpi-grid">{kpi_html}</div>

  <!-- ── Analytics trio ── -->
  <div class="section-heading">Metric Analysis</div>
  <div class="analytics-row">

    <!-- Gauges -->
    <div class="panel">
      <div class="panel-title">RAGAS Metric Scores</div>
      <div class="gauge-grid">{gauges_html}</div>
    </div>

    <!-- Faithfulness donut -->
    <div class="panel">
      <div class="panel-title">Faithfulness Distribution</div>
      <div style="display:flex;align-items:center;gap:24px;justify-content:center;flex-wrap:wrap">
        {donut_svg}
        <div>{faith_legend}</div>
      </div>
    </div>

    <!-- Difficulty breakdown -->
    <div class="panel">
      <div class="panel-title">Accuracy by Difficulty</div>
      {diff_html}
      <div style="margin-top:20px">
        <div class="panel-title" style="font-size:12px;margin-bottom:12px">Score Trends (per question)</div>
        {sparklines_html}
      </div>
    </div>

  </div>

  <!-- ── Detailed results table ── -->
  <div class="section-heading">Per-Question Results</div>
  <div class="table-wrap">
    <div class="table-header">
      <div>
        <div class="table-title">All Evaluation Questions</div>
        <div class="table-subtitle">Click any row to expand the full answer and diagnostics</div>
      </div>
    </div>
    <div class="filter-bar">
      <span class="filter-label">Filter:</span>
      <input class="filter-input" id="searchInput" placeholder="Search question or ID…"
        oninput="filterTable()">
      <select class="filter-select" id="diffFilter" onchange="filterTable()">
        <option value="">All Difficulties</option>
        <option value="easy">Easy</option>
        <option value="medium">Medium</option>
        <option value="hard">Hard</option>
      </select>
      <select class="filter-select" id="faithFilter" onchange="filterTable()">
        <option value="">All Faithfulness</option>
        <option value="fully_supported">Fully Supported</option>
        <option value="partially_supported">Partially Supported</option>
        <option value="not_supported">Not Supported</option>
        <option value="error">Error</option>
      </select>
      <select class="filter-select" id="typeFilter" onchange="filterTable()">
        <option value="">All Types</option>
        {"".join(f'<option value="{t}">{t.replace("_"," ").title()}</option>' for t in sorted(set(i.get("question_type","") for i in items)))}
      </select>
      <span class="filter-label" id="rowCount" style="margin-left:auto">{len(items)} rows</span>
    </div>
    <div style="overflow-x:auto">
    <table id="resultsTable">
      <thead>
        <tr>
          <th>ID</th>
          <th>Difficulty</th>
          <th>Type</th>
          <th>Question</th>
          <th>Faithfulness</th>
          <th>Ans. Relevancy</th>
          <th>Ctx. Precision</th>
          <th>Ctx. Recall</th>
          <th>Correctness</th>
          <th>Faith. Grade</th>
          <th>Retries</th>
          <th>Rewrites</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="tableBody">{table_rows}</tbody>
    </table>
    </div>
  </div>

  <footer class="footer">
    <div>PolicyIQ · Self-RAG Evaluation · Generated {ts_fmt}</div>
    <div>RAGAS · LangGraph · HuggingFace BGE · OpenAI GPT-4o-mini</div>
  </footer>

</div><!-- /page-wrap -->

<script>
  // ── Row expand/collapse ───────────────────────────────────────────────────
  function toggleDetail(idx) {{
    const row = document.getElementById('detail-' + idx);
    if (!row) return;
    row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
  }}

  // ── Table filter ─────────────────────────────────────────────────────────
  function filterTable() {{
    const search  = document.getElementById('searchInput').value.toLowerCase();
    const diff    = document.getElementById('diffFilter').value.toLowerCase();
    const faith   = document.getElementById('faithFilter').value.toLowerCase();
    const type    = document.getElementById('typeFilter').value.toLowerCase();
    const tbody   = document.getElementById('tableBody');
    const rows    = Array.from(tbody.rows).filter(r => !r.id.startsWith('detail-'));
    let visible   = 0;

    rows.forEach((row, i) => {{
      const text  = row.innerText.toLowerCase();
      const cells = row.cells;
      const idTxt   = cells[0]?.innerText.toLowerCase() || '';
      const diffTxt = cells[1]?.innerText.toLowerCase() || '';
      const typeTxt = cells[2]?.innerText.toLowerCase() || '';
      const qTxt    = cells[3]?.innerText.toLowerCase() || '';
      // faithfulness grade cell (index 9)
      const faithTxt = cells[9]?.innerText.toLowerCase() || '';

      const matchSearch = !search || idTxt.includes(search) || qTxt.includes(search);
      const matchDiff   = !diff  || diffTxt.includes(diff);
      const matchFaith  = !faith || faithTxt.includes(faith);
      const matchType   = !type  || typeTxt.includes(type);

      const show = matchSearch && matchDiff && matchFaith && matchType;
      row.style.display = show ? '' : 'none';

      // hide detail row if parent hidden
      const detailRow = document.getElementById('detail-' + i);
      if (detailRow && !show) detailRow.style.display = 'none';

      if (show) visible++;
    }});

    document.getElementById('rowCount').textContent = visible + ' rows';
  }}
</script>
</body>
</html>"""

    # ── Write file ───────────────────────────────────────────────────────────
    out_path = output_dir / f"self_rag_dashboard_{timestamp}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML dashboard written → {out_path}")
    return str(out_path)


# ── Standalone entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find the most recent report
        reports = sorted(Path("eval_results").glob("self_rag_report_*.json"), reverse=True)
        if not reports:
            print("Usage: python -m src.eval.generate_dashboard <path_to_report.json>")
            print("       or run from eval_results/ with at least one self_rag_report_*.json")
            sys.exit(1)
        report_path = reports[0]
        print(f"No path given — using most recent: {report_path}")
    else:
        report_path = Path(sys.argv[1])

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    ts  = data.get("timestamp", "20240101_000000")
    out = Path("eval_results")
    generate_dashboard(data, out, ts)