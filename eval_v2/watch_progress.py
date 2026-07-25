#!/usr/bin/env python3
"""
Progress reader for a running BEIR evaluation.

    # terminal
    python -m eval_v2.watch_progress
    python -m eval_v2.watch_progress --watch 60

    # publish a self-refreshing page to the HuggingFace Space
    python -m eval_v2.watch_progress --publish SajilAwale/Embedding-Compression-Eval-BEIR
    python -m eval_v2.watch_progress --publish <repo> --watch 600   # keep it fresh

`tail -f` on the SLURM log tells you what the job is doing *right now*, but not
how far through it is. This reads the durable state instead:

  * Encode progress  -- counts <cache_dir>/*/beir/<subset>/{corpus,queries}.npz.
    Encoding is the long pole (29 checkpoints x 13 subsets x 2 splits = 754
    files), and these appear as they complete.
  * Eval progress    -- counts (model, subset) pairs present in aggregate.json,
    which the evaluator appends to after every pair.
  * Liveness         -- last log line, and whether anything has been written
    recently. A job can sit "R" in squeue while wedged on a download.

Progress is reported per subset because BEIR subsets differ by three orders of
magnitude in corpus size (nfcorpus 3.6k docs vs msmarco 8.8M), so "42% of pairs
done" says almost nothing about remaining time. The per-subset view does.
"""
import argparse
import html as html_mod
import json
import os
import sys
import time
from pathlib import Path

_THESIS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_THESIS_ROOT))

from dotenv import load_dotenv
load_dotenv(_THESIS_ROOT / ".env")

EXT_DISK = Path("/nas/rgroup/dsig/llm-team/sajil-thesis")
STALL_SECONDS = 1800


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=str(EXT_DISK / "results/beir"))
    p.add_argument("--cache_dir", default=str(EXT_DISK / "eval_cache"))
    p.add_argument("--log", default=None,
                   help="SLURM .out to sample. Default: newest in eval_v2/slurm_logs/.")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="Refresh every N seconds instead of running once.")
    p.add_argument("--html", metavar="PATH",
                   help="Also write the snapshot as HTML to PATH.")
    p.add_argument("--publish", metavar="REPO_ID",
                   help="Upload the HTML to this HF Space. Implies --html.")
    p.add_argument("--publish_path", default="index.html",
                   help="Filename inside the Space. Default index.html, so the "
                        "Space URL itself renders the page (needed for a private "
                        "Space, which only renders its landing page in the HF UI). "
                        "Use progress.html to add it alongside an existing dashboard.")
    p.add_argument("--create", action="store_true",
                   help="Create the Space if it does not exist (static, private).")
    return p.parse_args()


def load_json(path: Path):
    """Tolerant read: the evaluator rewrites these files constantly, so a reader
    will eventually catch a partial write. Treat that as 'no data yet'."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def expected_scope():
    """(models, checkpoint paths, subsets) the full run covers -- same filter as
    eval_beir_full.sh, which skips training-snapshot checkpoints."""
    from eval_v2.config.models import MODELS
    from eval_v2.config.datasets import DATASETS
    models = {k: s for k, s in MODELS.items() if "/checkpoints/checkpoint-" not in s.path}
    paths = {s.path for s in models.values()}
    return models, paths, DATASETS["beir"].subsets


def newest_log() -> Path | None:
    d = _THESIS_ROOT / "eval_v2/slurm_logs"
    logs = sorted(d.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if d.is_dir() else []
    return logs[0] if logs else None


def collect(args) -> dict:
    models, paths, subsets = expected_scope()
    results, cache = Path(args.results_dir), Path(args.cache_dir)
    agg = load_json(results / "aggregate.json")

    rows = []
    for sub in subsets:
        n_enc = sum(1 for _ in cache.glob(f"*/beir/{sub}/*.npz")) if cache.is_dir() else 0
        n_eval = sum(1 for m in models if sub in agg.get(m, {}))
        rows.append({"subset": sub, "enc": n_enc, "enc_want": len(paths) * 2,
                     "ev": n_eval, "ev_want": len(models)})

    sig, index = results / "significance.json", results / "index.html"
    st = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": str(results), "rows": rows,
        "n_models": len(models), "n_ckpt": len(paths), "n_subsets": len(subsets),
        "enc_done": sum(r["enc"] for r in rows), "enc_total": sum(r["enc_want"] for r in rows),
        "ev_done": sum(r["ev"] for r in rows), "ev_total": sum(r["ev_want"] for r in rows),
        "sig": f"{sig.stat().st_size/1e6:.0f} MB" if sig.exists() else "not started",
        "index": "built" if index.exists() else "not built",
        "log": None,
    }

    log = Path(args.log) if args.log else newest_log()
    if log and log.is_file():
        age = time.time() - log.stat().st_mtime
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
        err = log.with_suffix(".err")
        st["log"] = {
            "name": log.name, "age_min": age / 60,
            "last": lines[-1][:120] if lines else "",
            "stalled": age > STALL_SECONDS,
            "tracebacks": err.read_text(errors="replace").count(
                "Traceback (most recent call last)") if err.is_file() else 0,
        }
    return st


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "-" * width
    return "#" * int(width * done / total) + "." * (width - int(width * done / total))


def render_text(st: dict) -> str:
    o = ["=" * 72,
         f"BEIR eval progress  {st['time']}",
         f"  results: {st['results']}",
         f"  scope:   {st['n_models']} model variants on {st['n_ckpt']} checkpoints "
         f"x {st['n_subsets']} subsets",
         "=" * 72,
         f"\n{'subset':18s} {'encode':>9s}  {'':28s} {'evaluated':>11s}"]
    for r in st["rows"]:
        partial = 0 < r["enc"] < r["enc_want"] or 0 < r["ev"] < r["ev_want"]
        o.append(f"{r['subset']:18s} {r['enc']:4d}/{r['enc_want']:<4d}  "
                 f"[{bar(r['ev'], r['ev_want'])}] {r['ev']:5d}/{r['ev_want']:<5d}"
                 f"{'  <- partial' if partial else ''}")
    pct = 100 * st["ev_done"] / max(st["ev_total"], 1)
    o += [f"\n{'TOTAL':18s} {st['enc_done']:4d}/{st['enc_total']:<4d}  "
          f"[{bar(st['ev_done'], st['ev_total'])}] "
          f"{st['ev_done']:5d}/{st['ev_total']:<5d} ({pct:.1f}%)",
          f"\nsignificance.json: {st['sig']}   index.html: {st['index']}"]
    if st["log"]:
        L = st["log"]
        o.append(f"\nlog: {L['name']}  (updated {L['age_min']:.1f} min ago)")
        o.append(f"  last: {L['last']}")
        if L["stalled"]:
            o.append("  WARNING: no log output for >30 min -- may be stalled.")
        if L["tracebacks"]:
            o.append(f"  {L['tracebacks']} traceback(s) in the .err file -- check it.")
    return "\n".join(o)


def render_html(st: dict, refresh: int = 120) -> str:
    e = html_mod.escape
    pct = 100 * st["ev_done"] / max(st["ev_total"], 1)

    def row(r):
        p = 100 * r["ev"] / max(r["ev_want"], 1)
        pe = 100 * r["enc"] / max(r["enc_want"], 1)
        done = r["ev"] == r["ev_want"]
        cls = "done" if done else ("active" if r["ev"] or r["enc"] else "idle")
        return f"""<tr class="{cls}">
  <td class="name">{e(r['subset'])}</td>
  <td class="num">{r['enc']}/{r['enc_want']}</td>
  <td class="barcell"><div class="track"><i style="width:{pe:.1f}%"></i></div></td>
  <td class="num">{r['ev']}/{r['ev_want']}</td>
  <td class="barcell"><div class="track"><i style="width:{p:.1f}%"></i></div></td>
</tr>"""

    warn = ""
    if st["log"] and st["log"]["stalled"]:
        warn = ('<p class="warn">No log output for over 30 minutes &mdash; '
                'the job may be stalled.</p>')
    if st["log"] and st["log"]["tracebacks"]:
        warn += (f'<p class="warn">{st["log"]["tracebacks"]} traceback(s) in the '
                 f'.err log.</p>')

    logblk = ""
    if st["log"]:
        logblk = (f'<p class="meta">log <code>{e(st["log"]["name"])}</code>, updated '
                  f'{st["log"]["age_min"]:.1f} min ago</p>'
                  f'<pre class="last">{e(st["log"]["last"])}</pre>')

    # Staleness is the failure mode that matters when nobody can reach the
    # server: if the publisher dies, this page just freezes at an old timestamp
    # and still looks plausible. Embed the generation epoch and let the viewer's
    # browser compute the age, so a dead publisher is visible from anywhere.
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>BEIR eval progress</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 2rem auto;
         max-width: 60rem; padding: 0 1rem; }}
  h1 {{ font-size: 1.25rem; margin-bottom: .25rem; }}
  .meta {{ opacity: .7; margin: .2rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1.25rem; }}
  th, td {{ padding: .35rem .5rem; text-align: left; border-bottom: 1px solid #8883; }}
  th {{ font-weight: 600; font-size: .8rem; text-transform: uppercase; opacity: .6; }}
  .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .name {{ font-weight: 600; }}
  .barcell {{ width: 30%; }}
  .track {{ background: #8882; border-radius: 3px; height: .55rem; overflow: hidden; }}
  .track i {{ display: block; height: 100%; background: #3b82f6; }}
  tr.done .track i {{ background: #22c55e; }}
  tr.idle {{ opacity: .55; }}
  .total {{ margin-top: 1.25rem; font-size: 1.05rem; }}
  .warn {{ background: #f591; border-left: 3px solid #f59e0b; padding: .5rem .75rem; }}
  pre.last {{ background: #8881; padding: .5rem .75rem; border-radius: 4px;
              overflow-x: auto; font-size: .8rem; }}
  .stale {{ background: #ef44441a; border-left: 3px solid #ef4444;
            padding: .5rem .75rem; font-weight: 600; }}
  @media (max-width: 40rem) {{ .barcell {{ display: none; }} }}
</style></head><body>
<h1>Full BEIR evaluation &mdash; progress</h1>
<div id="stale"></div>
<p class="meta"><span id="age">{e(st['time'])}</span> &middot; auto-refresh {refresh}s</p>
<script>
  // Published at this epoch (UTC seconds). If the publisher on dsig1 stops,
  // the page keeps serving but stops changing -- so compute the age client-side
  // and say so loudly rather than showing a stale snapshot as if it were live.
  var GEN = {int(time.time())}, EVERY = {refresh};
  function tick() {{
    var age = Math.floor(Date.now() / 1000 - GEN);
    var mins = Math.floor(age / 60);
    document.getElementById('age').textContent =
      'generated ' + (mins < 1 ? 'just now' : mins + ' min ago') +
      ' ({e(st['time'])})';
    document.getElementById('stale').innerHTML = (age > EVERY * 3)
      ? '<p class="stale">This page has not updated in ' + mins +
        ' minutes. The publisher may have stopped &mdash; these numbers are ' +
        'not current.</p>' : '';
  }}
  tick(); setInterval(tick, 30000);
</script>
<p class="meta">{st['n_models']} model variants on {st['n_ckpt']} checkpoints
   &times; {st['n_subsets']} subsets</p>
{warn}
<table>
  <thead><tr><th>subset</th><th>encoded</th><th></th><th>evaluated</th><th></th></tr></thead>
  <tbody>
{chr(10).join(row(r) for r in st['rows'])}
  </tbody>
</table>
<p class="total"><strong>{st['ev_done']} / {st['ev_total']}</strong>
   model&times;subset pairs evaluated ({pct:.1f}%) &middot;
   {st['enc_done']} / {st['enc_total']} encodes</p>
<p class="meta">significance.json: {e(st['sig'])} &middot; dashboard: {e(st['index'])}</p>
{logblk}
</body></html>"""


def publish(repo_id: str, html_path: Path, path_in_repo: str, create: bool):
    from huggingface_hub import HfApi, create_repo
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("  publish skipped: HUGGINGFACE_TOKEN not set", file=sys.stderr)
        return
    if create:
        create_repo(repo_id, repo_type="space", space_sdk="static",
                    private=True, exist_ok=True, token=token)
    HfApi(token=token).upload_file(
        path_or_fileobj=str(html_path), path_in_repo=path_in_repo,
        repo_id=repo_id, repo_type="space", commit_message="progress update")
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"  published -> {url}" if path_in_repo == "index.html"
          else f"  published -> {url}/blob/main/{path_in_repo}")


def once(args):
    st = collect(args)
    print(render_text(st))
    out = Path(args.html) if args.html else None
    if args.publish and out is None:
        out = Path("/tmp/beir_progress.html")
    if out:
        out.write_text(render_html(st, refresh=max(args.watch or 120, 30)))
        print(f"\n  wrote {out}")
    if args.publish:
        publish(args.publish, out, args.publish_path, args.create)


def main():
    args = parse_args()
    if not args.watch:
        once(args)
        return
    try:
        while True:
            if not args.publish:
                os.system("clear")
            once(args)
            print(f"\n(refreshing every {args.watch}s -- Ctrl-C to stop)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
