#!/usr/bin/env python3
"""Generate a self-contained interactive HTML dashboard from eval result JSONs.

Can be run standalone:
    python interactive_plot.py --dataset_name nanobeir ...

Or imported and called from eval.py:
    from interactive_plot import generate_interactive_html
"""

import argparse
import json
import os

# ---------------------------------------------------------------------------
# Model-group helpers (mirrors eval.py logic, no import needed)
# ---------------------------------------------------------------------------

def _classify_group(key: str) -> str:
    k = key.lower()
    if "mrl" in k and "bat" in k:
        return "MRL+BAT"
    if "mrl" in k and "pq" in k:
        return "MRL+PQ"
    if "mrl" in k and ("post-binary" in k or ("bin" in k and "pq" not in k and "bat" not in k)):
        return "MRL+Bin"
    if "mrl" in k:
        return "MRL"
    if "bat" in k:
        return "BAT"
    if "pq" in k:
        return "PQ"
    if "post-int8" in k or "post-binary" in k:
        return "Post-Quant"
    if "tq" in k:
        return "TurboQuant"
    return "Baseline"


def _classify_marker(key: str) -> str:
    """Mirror eval.py's _get_marker_for_model, mapped to Plotly symbol names."""
    k = key.lower()
    if "bat" in k:
        return "diamond"          # D
    if "mrl" in k and "post-binary" in k:
        return "triangle-down"    # v
    if "post-int8" in k or ("post-binary" in k and "mrl" not in k):
        return "square"           # s
    if "mrl" in k and "pq" in k:
        return "star"             # *
    if "pq" in k:
        return "cross"            # P  (plus-filled → cross is closest in Plotly)
    if "mrl" in k:
        return "triangle-up"      # ^
    if "tq" in k:
        return "hexagon"          # h
    return "circle"               # o


# Matplotlib hatch syntax (as stored in eval.py models dict) → Plotly fillpattern shape
_HATCH_TO_PLOTLY = {
    "//":   "/",
    "\\\\":  "\\",   # eval.py stores "\\\\" which is the string \\
    "xx":   "x",
    "--":   "-",
    "||":   "|",
    "++":   "+",
    "oo":   ".",
    "**":   "x",    # no direct star pattern in Plotly; x is closest
    "OO":   ".",    # TurboQuant — circles/dots
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_payload(dataset_name: str, raw_results: dict, raw_emb_info: dict,
                   models_meta: dict) -> dict:
    """Convert raw JSON dumps into the compact structure consumed by the HTML."""

    # --- results: model -> subset -> metric -> k(str) -> float ---
    results = {}
    for model_key, metrics_dict in raw_results.items():
        model_res = {}
        for full_key, value in metrics_dict.items():
            # format: dataset__subset____evaluator_simfn_metric@k
            parts = full_key.split("__")
            if len(parts) < 3:
                continue
            subset = parts[1]
            tail = parts[-1]   # e.g. "evaluator_cosine_mrr@1"
            if "@" not in tail:
                continue
            metric_part, k_str = tail.rsplit("@", 1)
            metric = metric_part.split("_")[-1]
            if subset not in model_res:
                model_res[subset] = {}
            if metric not in model_res[subset]:
                model_res[subset][metric] = {}
            model_res[subset][metric][k_str] = round(float(value), 6)
        if model_res:
            results[model_key] = model_res

    # --- emb_info: model -> subset -> size details ---
    emb_info = {}
    for model_key, eval_dict in raw_emb_info.items():
        model_emb = {}
        for eval_name, info in eval_dict.items():
            parts = eval_name.split("__")
            if len(parts) < 2:
                continue
            subset = parts[1]
            queries = info.get("queries", {})
            corpus  = info.get("corpus",  {})
            q_bits  = queries.get("n", 0) * queries.get("dimension", 0) * queries.get("element_size_bit", 0)
            c_bits  = corpus.get("n",  0) * corpus.get("dimension",  0) * corpus.get("element_size_bit",  0)
            model_emb[subset] = {
                "total_mb":           round((q_bits + c_bits) / (8 * 1024 * 1024), 6),
                "corpus_mb":          round(c_bits / (8 * 1024 * 1024), 6),
                "query_dim":          queries.get("dimension", 0),
                "corpus_dim":         corpus.get("dimension",  0),
                "query_bits_per_el":  queries.get("element_size_bit", 0),
                "corpus_bits_per_el": corpus.get("element_size_bit",  0),
                "n_queries":          queries.get("n", 0),
                "n_corpus":           corpus.get("n",  0),
            }
        if model_emb:
            emb_info[model_key] = model_emb

    # --- model metadata ---
    all_keys = sorted(set(results) | set(emb_info))
    meta = {}
    for key in all_keys:
        user = models_meta.get(key, {})
        meta[key] = {
            "display_name": user.get("display_name", key),
            "color":        user.get("color", None),
            "group":        _classify_group(key),
            "marker":       _classify_marker(key),
            "hatch":        _HATCH_TO_PLOTLY.get(user.get("hatch", ""), ""),
        }

    return {
        "dataset": dataset_name,
        "models_meta": meta,
        "results":  results,
        "emb_info": emb_info,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Eval Explorer — {dataset}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f0f2f5; color: #222; font-size: 13px;
}}
#app {{ display: flex; height: 100vh; overflow: hidden; }}

/* ---- Sidebar ---- */
#sidebar {{
  width: 240px; min-width: 200px; flex-shrink: 0;
  background: #fff; border-right: 1px solid #dde1e7;
  padding: 14px 12px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 12px;
}}
#sidebar h2 {{ font-size: 14px; font-weight: 700; color: #444; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
.ctrl {{ display: flex; flex-direction: column; gap: 3px; }}
.ctrl label {{
  font-size: 10px; font-weight: 700; color: #888;
  text-transform: uppercase; letter-spacing: 0.6px;
}}
.ctrl select, .ctrl input[type=range] {{
  padding: 5px 7px; border: 1px solid #ddd; border-radius: 5px;
  font-size: 12px; background: #fafafa; width: 100%;
  cursor: pointer;
}}
.ctrl select:focus {{ outline: 2px solid #4a90d9; border-color: transparent; }}
.chk-grid {{ display: flex; flex-direction: column; gap: 3px; }}
.chk-item {{ display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 2px 0; }}
.chk-item input {{ cursor: pointer; accent-color: #4a90d9; }}
.chk-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.chk-label {{ font-size: 12px; }}
hr.divider {{ border: none; border-top: 1px solid #eee; }}

/* ---- Main ---- */
#main {{ flex: 1; display: flex; flex-direction: column; min-height: 0; }}
#plot-wrap {{ flex: 1; padding: 10px 12px 0; min-height: 0; overflow: auto; position: relative; }}
#plotly-chart {{ width: 100%; height: 100%; }}

/* ---- Table ---- */
#table-wrap {{
  flex-shrink: 0;
  border-top: 1px solid #dde1e7;
  height: 180px; overflow: auto;
  padding: 0 12px 8px;
  background: #fff;
}}
#data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
#data-table th {{
  position: sticky; top: 0; background: #f5f7fa;
  padding: 5px 8px; text-align: left;
  border-bottom: 2px solid #dde1e7; font-weight: 600; color: #555;
  white-space: nowrap;
}}
#data-table td {{ padding: 3px 8px; border-bottom: 1px solid #f0f0f0; }}
#data-table tr:hover td {{ background: #f0f7ff; }}
.tbl-num {{ font-variant-numeric: tabular-nums; }}
.tbl-sn  {{ color: #aaa; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<div id="app">

  <!-- ===== SIDEBAR ===== -->
  <div id="sidebar">
    <h2>Eval Explorer</h2>

    <div class="ctrl">
      <label>Plot type</label>
      <select id="sel-type">
        <option value="bar">Bar — model comparison</option>
        <option value="scatter">Scatter — perf vs size</option>
        <option value="heatmap">Heatmap — models × subsets</option>
        <option value="line_k">Line — metric over k</option>
      </select>
    </div>

    <div class="ctrl">
      <label>Metric</label>
      <select id="sel-metric"></select>
    </div>

    <div class="ctrl" id="ctrl-k">
      <label>@ k</label>
      <select id="sel-k"></select>
    </div>

    <div class="ctrl" id="ctrl-subset">
      <label>Subset</label>
      <select id="sel-subset"></select>
    </div>

    <div class="ctrl" id="ctrl-size" style="display:none">
      <label>Size axis</label>
      <select id="sel-size">
        <option value="total_mb">Total (queries + corpus)</option>
        <option value="corpus_mb">Corpus only</option>
      </select>
    </div>

    <hr class="divider"/>

    <div class="ctrl">
      <label>Model groups</label>
      <div class="chk-grid" id="group-checks"></div>
    </div>
  </div>

  <!-- ===== MAIN ===== -->
  <div id="main">
    <div id="plot-wrap">
      <div id="plotly-chart"></div>
    </div>
    <div id="table-wrap">
      <table id="data-table">
        <thead><tr id="tbl-head"></tr></thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
// ============================================================
// Embedded data
// ============================================================
const DATA = __PAYLOAD__;

// ============================================================
// Constants
// ============================================================
const GROUP_ORDER  = ["Baseline","Post-Quant","BAT","PQ","TurboQuant","MRL","MRL+Bin","MRL+PQ","MRL+BAT"];
const GROUP_COLORS = {{
  "Baseline":   "#1f77b4",
  "Post-Quant": "#756bb1",
  "BAT":        "#8c564b",
  "MRL":        "#31a354",
  "MRL+Bin":    "#f16913",
  "MRL+PQ":     "#0096c7",
  "MRL+BAT":    "#e91e8c",
  "PQ":         "#c9184a",
  "TurboQuant": "#d4a017",
}};
const METRIC_LABELS = {{
  mrr: "MRR", ndcg: "nDCG", accuracy: "Accuracy",
  precision: "Precision", recall: "Recall", map: "MAP",
}};
const METRIC_ORDER = ["mrr","ndcg","accuracy","precision","recall","map"];

// ============================================================
// Helpers
// ============================================================
const $ = id => document.getElementById(id);

function modelColor(key) {{
  const m = DATA.models_meta[key];
  return m.color || GROUP_COLORS[m.group] || "#888";
}}

function modelMarker(key) {{
  const m = DATA.models_meta[key]?.marker;
  // Pass through any valid Plotly symbol name; fallback to circle
  return m || "circle";
}}

function dname(key) {{ return DATA.models_meta[key]?.display_name || key; }}

function metricLabel(m) {{ return METRIC_LABELS[m] || m.toUpperCase(); }}

function setOptions(selId, opts) {{
  const sel = $(selId);
  const prev = sel.value;
  sel.innerHTML = "";
  opts.forEach(([v, lbl]) => {{
    const o = document.createElement("option");
    o.value = v; o.textContent = lbl; sel.appendChild(o);
  }});
  if (opts.some(([v]) => v === prev)) sel.value = prev;
}}

function getActiveModels() {{
  const checked = new Set(
    [...document.querySelectorAll("#group-checks input:checked")].map(e => e.value)
  );
  return Object.keys(DATA.models_meta).filter(k =>
    checked.has(DATA.models_meta[k].group) && DATA.results[k]
  );
}}

function orderedModels(keys) {{
  const g = {{}};
  keys.forEach(k => {{
    const grp = DATA.models_meta[k].group;
    (g[grp] = g[grp] || []).push(k);
  }});
  return GROUP_ORDER.flatMap(grp => g[grp] || []);
}}

// ============================================================
// Init
// ============================================================
function init() {{
  // Collect available metrics / ks / subsets from results
  const allMetrics = new Set(), allKs = new Set(), allSubsets = new Set();
  Object.values(DATA.results).forEach(mRes =>
    Object.entries(mRes).forEach(([subset, metricDict]) => {{
      allSubsets.add(subset);
      Object.entries(metricDict).forEach(([metric, kDict]) => {{
        allMetrics.add(metric);
        Object.keys(kDict).forEach(k => allKs.add(parseInt(k)));
      }});
    }})
  );

  // Metric selector
  const sortedMetrics = METRIC_ORDER.filter(m => allMetrics.has(m))
    .concat([...allMetrics].filter(m => !METRIC_ORDER.includes(m)).sort());
  setOptions("sel-metric", sortedMetrics.map(m => [m, metricLabel(m)]));

  // K selector
  const sortedKs = [...allKs].sort((a,b) => a-b);
  setOptions("sel-k", sortedKs.map(k => [String(k), `@ ${k}`]));

  // Subset selector — mean/weightedmean first
  const prioritySubsets = ["mean","weightedmean"].filter(s => allSubsets.has(s));
  const otherSubsets    = [...allSubsets].filter(s => !["mean","weightedmean"].includes(s)).sort();
  setOptions("sel-subset", [...prioritySubsets, ...otherSubsets].map(s => [s, s]));

  // Group checkboxes
  const container = $("group-checks");
  container.innerHTML = "";
  GROUP_ORDER.forEach(grp => {{
    if (!Object.values(DATA.models_meta).some(m => m.group === grp)) return;
    const label = document.createElement("label");
    label.className = "chk-item";
    label.innerHTML = `
      <input type="checkbox" value="${{grp}}" checked>
      <span class="chk-dot" style="background:${{GROUP_COLORS[grp]}}"></span>
      <span class="chk-label">${{grp}}</span>`;
    container.appendChild(label);
  }});

  // Wire events
  ["sel-type","sel-metric","sel-k","sel-subset","sel-size"].forEach(id =>
    $(id).addEventListener("change", onControlChange)
  );
  container.addEventListener("change", buildPlot);

  $("sel-type").addEventListener("change", function() {{
    const pt = this.value;
    $("ctrl-k").style.display      = pt === "line_k"  ? "none" : "";
    $("ctrl-subset").style.display = pt === "heatmap" ? "none" : "";
    $("ctrl-size").style.display   = pt === "scatter" ? ""     : "none";
    $("table-wrap").style.display  = pt === "heatmap" ? "none" : "";
    buildPlot();
  }});

  buildPlot();
}}

function onControlChange() {{ buildPlot(); }}

// ============================================================
// Dispatch
// ============================================================
function buildPlot() {{
  const pt     = $("sel-type").value;
  const metric = $("sel-metric").value;
  const k      = $("sel-k").value;
  const subset = $("sel-subset").value;
  const models = getActiveModels();

  if      (pt === "bar")     plotBar(models, subset, metric, k);
  else if (pt === "scatter") plotScatter(models, subset, metric, k, $("sel-size").value);
  else if (pt === "heatmap") plotHeatmap(models, metric, k);
  else if (pt === "line_k")  plotLineK(models, subset, metric);
}}

// ============================================================
// Bar
// ============================================================
function plotBar(models, subset, metric, k) {{
  const ordered = [...models].sort((a, b) => dname(a).localeCompare(dname(b)));
  const xs = [], ys = [], colors = [], patterns = [], texts = [], hovers = [];
  const tableRows = [];

  ordered.forEach(key => {{
    const val = DATA.results[key]?.[subset]?.[metric]?.[k];
    if (val === undefined) return;
    xs.push(dname(key));
    ys.push(val);
    colors.push(modelColor(key));
    patterns.push(DATA.models_meta[key]?.hatch ?? "");
    texts.push(val.toFixed(3));
    hovers.push(`<b>${{dname(key)}}</b><br>Group: ${{DATA.models_meta[key].group}}<br>${{metricLabel(metric)}}@${{k}}: ${{val.toFixed(4)}}`);
    tableRows.push({{ Model: dname(key), Group: DATA.models_meta[key].group, [`${{metricLabel(metric)}}@${{k}}`]: val.toFixed(4) }});
  }});

  _plotIndex = {{}};
  xs.forEach((name, i) => {{ _plotIndex[name] = {{curve: 0, point: i}}; }});

  const trace = {{
    type: "bar", x: xs, y: ys,
    marker: {{
      color: colors,
      pattern: {{ shape: patterns, fgcolor: "rgba(0,0,0,0.45)", size: 6 }},
    }},
    text: texts, textposition: "outside",
    hovertemplate: hovers.map(h => h + "<extra></extra>"),
    cliponaxis: false,
  }};

  Plotly.react("plotly-chart", [trace], {{
    title: {{ text: `${{metricLabel(metric)}} @ ${{k}}  ·  ${{subset}}`, font: {{ size: 14 }} }},
    xaxis: {{ tickangle: -40, automargin: true, categoryorder: "array", categoryarray: xs }},
    yaxis: {{ title: `${{metricLabel(metric)}} @ ${{k}}`, range: [0, Math.min(1, Math.max(...ys) * 1.18)] }},
    showlegend: false,
    plot_bgcolor: "white", paper_bgcolor: "white",
    margin: {{ t: 50, l: 60, r: 20, b: 130 }},
  }}, {{ responsive: true }});

  renderTable(tableRows);
}}

// ============================================================
// Scatter — perf vs size
// ============================================================
function plotScatter(models, subset, metric, k, sizeKey) {{
  const byGroup = {{}};
  const tableRows = [];

  models.forEach(key => {{
    const val  = DATA.results[key]?.[subset]?.[metric]?.[k];
    const emb  = DATA.emb_info[key]?.[subset];
    if (val === undefined || !emb) return;
    const sizeMB = emb[sizeKey] ?? emb.total_mb;
    const grp = DATA.models_meta[key].group;
    (byGroup[grp] = byGroup[grp] || {{ x:[], y:[], text:[], keys:[] }});
    byGroup[grp].x.push(sizeMB);
    byGroup[grp].y.push(val);
    byGroup[grp].text.push(dname(key));
    byGroup[grp].keys.push(key);
    const embInfo = DATA.emb_info[key]?.[subset] || {{}};
    const bitsPerEmb = (embInfo.corpus_dim || 0) * (embInfo.corpus_bits_per_el || 0);
    tableRows.push({{
      Model: dname(key), Group: grp,
      "Size (MB)": sizeMB.toFixed(3),
      [`${{metricLabel(metric)}}@${{k}}`]: val.toFixed(4),
      "Bits/embedding": bitsPerEmb || "—",
    }});
  }});

  const traces = GROUP_ORDER.filter(g => byGroup[g]).map(grp => ({{
    type: "scatter", mode: "markers",
    name: grp,
    x: byGroup[grp].x, y: byGroup[grp].y,
    marker: {{
      size: 11,
      color: byGroup[grp].keys.map(k => modelColor(k)),
      symbol: byGroup[grp].keys.map(k => modelMarker(k)),
      line: {{ width: 1, color: "rgba(0,0,0,0.25)" }},
    }},
    hovertemplate: byGroup[grp].keys.map((key, i) =>
      `<b>${{dname(key)}}</b><br>Size: ${{byGroup[grp].x[i].toFixed(3)}} MB<br>${{metricLabel(metric)}}@${{k}}: ${{byGroup[grp].y[i].toFixed(4)}}<extra></extra>`
    ),
  }}));

  _plotIndex = {{}};
  traces.forEach((tr, ti) => {{
    byGroup[tr.name].text.forEach((name, pi) => {{ _plotIndex[name] = {{curve: ti, point: pi}}; }});
  }});

  const xLabel = sizeKey === "corpus_mb" ? "Corpus Embedding Size (MB)" : "Total Embedding Size (MB)";
  Plotly.react("plotly-chart", traces, {{
    title: {{ text: `${{metricLabel(metric)}}@${{k}}  vs  Embedding Size  ·  ${{subset}}`, font:{{ size:14 }} }},
    xaxis: {{ title: xLabel }},
    yaxis: {{ title: `${{metricLabel(metric)}} @ ${{k}}` }},
    legend: {{ title: {{ text: "Group" }}, font: {{ size: 10 }} }},
    hovermode: "closest",
    plot_bgcolor: "white", paper_bgcolor: "white",
    margin: {{ t: 50, l: 60, r: 20, b: 60 }},
  }}, {{ responsive: true }});

  renderTable(tableRows);
}}

// ============================================================
// Heatmap — models × subsets
// ============================================================
function plotHeatmap(models, metric, k) {{
  const allSubsets = new Set();
  models.forEach(key =>
    Object.keys(DATA.results[key] || {{}}).forEach(s => {{
      if (s !== "mean" && s !== "weightedmean") allSubsets.add(s);
    }})
  );
  // add mean/weightedmean at start
  const subsets = ["mean","weightedmean"].filter(s => allSubsets.has(s) || models.some(key => DATA.results[key]?.[s]))
    .concat([...allSubsets].sort());

  const ordered = orderedModels(models);
  const z = [], yLabels = [], tableRows = [];

  ordered.forEach(key => {{
    const row = subsets.map(s => DATA.results[key]?.[s]?.[metric]?.[k] ?? null);
    if (row.some(v => v !== null)) {{
      z.push(row);
      yLabels.push(dname(key));
      const rowObj = {{ Model: dname(key) }};
      subsets.forEach((s, i) => {{ rowObj[s] = row[i] !== null ? row[i].toFixed(4) : "—"; }});
      tableRows.push(rowObj);
    }}
  }});

  Plotly.react("plotly-chart", [{{
    type: "heatmap",
    z, x: subsets, y: yLabels,
    colorscale: "RdYlGn",
    zmid: 0.5,
    hovertemplate: "<b>%{{y}}</b><br>%{{x}}<br>" + metricLabel(metric) + `@${{k}}: %{{z:.4f}}<extra></extra>`,
    colorbar: {{ title: {{ text: `${{metricLabel(metric)}}@${{k}}`, side: "right" }}, len: 0.8 }},
  }}], {{
    title: {{ text: `${{metricLabel(metric)}} @ ${{k}}  ·  All Subsets`, font: {{ size: 14 }} }},
    xaxis: {{ tickangle: -40, automargin: true }},
    yaxis: {{ automargin: true }},
    plot_bgcolor: "white", paper_bgcolor: "white",
    margin: {{ t: 50, l: 220, r: 80, b: 160 }},
    height: Math.max(500, yLabels.length * 26 + 260),
  }}, {{ responsive: false }});

  renderTable(tableRows);
}}

// ============================================================
// Line — metric over k
// ============================================================
function plotLineK(models, subset, metric) {{
  const allKs = new Set();
  models.forEach(key =>
    Object.keys(DATA.results[key]?.[subset]?.[metric] || {{}}).forEach(k => allKs.add(parseInt(k)))
  );
  const sortedKs = [...allKs].sort((a,b) => a-b);

  const ordered = orderedModels(models);
  const traces = [], tableRows = [];

  ordered.forEach(key => {{
    const kDict = DATA.results[key]?.[subset]?.[metric];
    if (!kDict) return;
    const ys = sortedKs.map(k => kDict[String(k)] ?? null);
    if (ys.every(v => v === null)) return;
    traces.push({{
      type: "scatter", mode: "lines+markers",
      name: dname(key),
      x: sortedKs, y: ys,
      line:   {{ color: modelColor(key), width: 2 }},
      marker: {{ color: modelColor(key), size: 7, symbol: modelMarker(key) }},
      hovertemplate: `<b>${{dname(key)}}</b><br>k=%{{x}}<br>${{metricLabel(metric)}}: %{{y:.4f}}<extra></extra>`,
    }});
    const rowObj = {{ Model: dname(key), Group: DATA.models_meta[key].group }};
    sortedKs.forEach(k => {{ rowObj[`@${{k}}`] = kDict[String(k)]?.toFixed(4) ?? "—"; }});
    tableRows.push(rowObj);
  }});

  _plotIndex = {{}};
  traces.forEach((tr, ti) => {{ _plotIndex[tr.name] = {{curve: ti, point: 0}}; }});

  Plotly.react("plotly-chart", traces, {{
    title: {{ text: `${{metricLabel(metric)}}  over k  ·  ${{subset}}`, font: {{ size: 14 }} }},
    xaxis: {{ title: "k", tickvals: sortedKs, dtick: 1 }},
    yaxis: {{ title: metricLabel(metric) }},
    legend: {{ font: {{ size: 10 }}, tracegroupgap: 2 }},
    hovermode: "x unified",
    plot_bgcolor: "white", paper_bgcolor: "white",
    margin: {{ t: 50, l: 60, r: 180, b: 60 }},
  }}, {{ responsive: true }});

  renderTable(tableRows);
}}

// ============================================================
// Table renderer  (sortable)
// ============================================================
let _tableRows = [];
let _sortCol   = null;
let _sortDir   = 0;  // 0=none, 1=asc, -1=desc
let _plotIndex = {};  // displayName -> {{curve, point}}

function renderTable(rows) {{
  _tableRows = rows;
  _sortCol   = null;
  _sortDir   = 0;
  _drawTable();
}}

function _drawTable() {{
  if (!_tableRows.length) {{
    $("tbl-head").innerHTML = "";
    $("tbl-body").innerHTML = "";
    return;
  }}
  const cols = Object.keys(_tableRows[0]);

  let rows = _tableRows;
  if (_sortCol !== null && _sortDir !== 0) {{
    rows = [..._tableRows].sort((a, b) => {{
      const av = a[cols[_sortCol]], bv = b[cols[_sortCol]];
      const an = parseFloat(av), bn = parseFloat(bv);
      const aVal = !isNaN(an) && av !== "—" ? an : String(av ?? "");
      const bVal = !isNaN(bn) && bv !== "—" ? bn : String(bv ?? "");
      if (aVal < bVal) return -_sortDir;
      if (aVal > bVal) return _sortDir;
      return 0;
    }});
  }}

  // SN header (not sortable)
  const snTh = `<th style="color:#aaa;white-space:nowrap">#</th>`;
  $("tbl-head").innerHTML = snTh + cols.map((c, i) => {{
    let arrow = (_sortCol === i) ? (_sortDir === 1 ? " ▲" : _sortDir === -1 ? " ▼" : "") : "";
    return `<th style="cursor:pointer;user-select:none" onclick="sortTable(${{i}})">${{c}}${{arrow}}</th>`;
  }}).join("");

  $("tbl-body").innerHTML = rows.map((row, ri) => {{
    const safeName = String(row.Model ?? "").replace(/"/g, "&quot;");
    return `<tr data-name="${{safeName}}" onmouseenter="hlRow(this)" onmouseleave="unhlRow()">` +
      `<td class="tbl-sn">${{ri + 1}}</td>` +
      cols.map(c => {{
        const v = row[c];
        return `<td>${{v ?? "—"}}</td>`;
      }}).join("") +
      `</tr>`;
  }}).join("");
}}

function hlRow(tr) {{
  const name = tr.dataset.name;
  const idx  = _plotIndex[name];
  if (idx == null) return;
  try {{
    Plotly.Fx.hover($("plotly-chart"), [{{curveNumber: idx.curve, pointNumber: idx.point}}]);
  }} catch(e) {{}}
}}

function unhlRow() {{
  try {{ Plotly.Fx.unhover($("plotly-chart")); }} catch(e) {{}}
}}

function sortTable(colIdx) {{
  if (_sortCol === colIdx) {{
    if      (_sortDir === 0)  {{ _sortDir =  1; }}
    else if (_sortDir === 1)  {{ _sortDir = -1; }}
    else                      {{ _sortDir =  0; _sortCol = null; }}
  }} else {{
    _sortCol = colIdx;
    _sortDir = 1;
  }}
  _drawTable();
}}

// ============================================================
// Boot
// ============================================================
window.addEventListener("load", init);
window.addEventListener("resize", () => Plotly.Plots.resize("plotly-chart"));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_interactive_html(
    json_output_path: str,
    emb_info_path: str,
    dataset_name: str,
    output_dir: str,
    models_meta: dict | None = None,
) -> str:
    """Build the interactive HTML file and return its path.

    Parameters
    ----------
    json_output_path : path to the eval dump JSON  (e.g. results_json/nanobeir_eval_dump.json)
    emb_info_path    : path to the emb-info JSON    (e.g. results_emb_info/nanobeir_emb_info.json)
    dataset_name     : e.g. "nanobeir"
    output_dir       : directory in which to write the HTML file
    models_meta      : optional dict {model_key: {display_name, color, ...}} from eval.py's models dict
    """
    raw_results  = _load_json(json_output_path)
    raw_emb_info = _load_json(emb_info_path)

    if not raw_results:
        print(f"[interactive_plot] No results found at {json_output_path}, skipping HTML generation.")
        return ""

    payload = _build_payload(
        dataset_name, raw_results, raw_emb_info,
        models_meta or {},
    )

    # The template uses {{ / }} escaping for Python .format() compatibility;
    # unescape those first, then substitute sentinels.
    html = (
        _HTML_TEMPLATE
        .replace("{dataset}", "__DATASET__")   # in case old marker still present
        .replace("{{", "{").replace("}}", "}")
        .replace("__DATASET__", dataset_name)
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[interactive_plot] Saved interactive dashboard → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive eval HTML dashboard.")
    parser.add_argument("--dataset_name",         type=str, default="nanobeir")
    parser.add_argument("--json_output_path",      type=str, default="results_json/")
    parser.add_argument("--emb_info_path",         type=str, default="results_emb_info/")
    parser.add_argument("--interactive_plot_path", type=str, default="result_interactive_plot/")
    args = parser.parse_args()

    # Resolve file paths (allow passing either a dir or a direct .json path)
    json_path = args.json_output_path
    if os.path.isdir(json_path):
        json_path = os.path.join(json_path, f"{args.dataset_name}_eval_dump.json")

    emb_path = args.emb_info_path
    if os.path.isdir(emb_path):
        emb_path = os.path.join(emb_path, f"{args.dataset_name}_emb_info.json")

    generate_interactive_html(
        json_output_path=json_path,
        emb_info_path=emb_path,
        dataset_name=args.dataset_name,
        output_dir=args.interactive_plot_path,
    )
