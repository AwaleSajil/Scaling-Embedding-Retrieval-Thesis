#!/usr/bin/env python3
"""
Deploy an eval_v2 dashboard to a HuggingFace static Space.

Generalises the original hf_deploy.py (which hardcoded the NanoBEIR repo and
folder) so the same code publishes any dataset's results:

    python -m eval_v2.hf_deploy_dashboard \
        --results_dir /nas/rgroup/dsig/llm-team/sajil-thesis/smoke_matrix/results/beir \
        --repo_id     SajilAwale/Embedding-Compression-Eval-BEIR \
        --private --dry_run

What actually gets uploaded: index.html plus the two shard directories it
fetches at runtime (`./per_query_shards/<subset>.json` and
`./significance_shards/<file>.json`). Nothing else is needed -- notably NOT the
float32 embedding cache, which is an intermediate the evaluator consumes, and
not the monolithic aggregate/per_query/significance JSONs, which the dashboard
does not read.

plotly.min.js is a special case: index.html loads it locally with a CDN
fallback, but the eval pipeline never writes it. It is copied from an existing
results directory (--plotly_src) when missing.
"""
import argparse
import shutil
import sys
from pathlib import Path

_THESIS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_THESIS_ROOT))

from dotenv import load_dotenv
load_dotenv(_THESIS_ROOT / ".env")

import os

# Files the static Space needs. Order matters only for the dry-run printout.
ALLOW_PATTERNS = [
    "index.html",
    "README.md",
    "plotly.min.js",
    "per_query_shards/*",
    "significance_shards/*",
]

# HuggingFace rejects single files above this. The msmarco per-query shard is
# the one at risk on full BEIR (~1.5 GB projected), so warn well before it.
HF_MAX_FILE_BYTES = 50 * 1000**3
WARN_FILE_BYTES = 500 * 1000**2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True,
                   help="Dataset results dir, i.e. <output_dir>/<dataset> "
                        "(the one containing index.html).")
    p.add_argument("--repo_id", required=True,
                   help="Target Space, e.g. SajilAwale/Embedding-Compression-Eval-BEIR")
    p.add_argument("--private", action="store_true",
                   help="Create the Space private. Existing repos keep their current "
                        "visibility -- this only applies at creation.")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be uploaded, with sizes, and exit. "
                        "Creates nothing.")
    p.add_argument("--plotly_src",
                   default=str(_THESIS_ROOT / "eval_v2/outputs/results/nanobeir/plotly.min.js"),
                   help="Where to copy plotly.min.js from if results_dir lacks it.")
    p.add_argument("--commit_message", default="Deploy eval dashboard")
    return p.parse_args()


def collect(results: Path) -> list[Path]:
    """Files matching ALLOW_PATTERNS, in upload order."""
    out: list[Path] = []
    for pat in ALLOW_PATTERNS:
        out.extend(sorted(results.glob(pat)))
    return out


def main():
    args = parse_args()
    results = Path(args.results_dir).resolve()

    if not results.is_dir():
        sys.exit(f"ERROR: results_dir does not exist: {results}")
    index = results / "index.html"
    if not index.is_file():
        sys.exit(f"ERROR: no index.html in {results}\n"
                 f"       Run the eval first, or `python -m eval_v2.run --just_html`.")

    # index.html fetches the shard dirs at runtime; without them the Space loads
    # but every subset and significance panel silently comes up empty.
    for d in ("per_query_shards", "significance_shards"):
        if not (results / d).is_dir():
            print(f"WARNING: {d}/ missing -- the dashboard will render empty panels.")

    # plotly.min.js is not produced by the eval; supply it or rely on the CDN.
    dest_plotly = results / "plotly.min.js"
    if not dest_plotly.exists():
        src_plotly = Path(args.plotly_src)
        if src_plotly.is_file():
            if args.dry_run:
                print(f"[dry-run] would copy {src_plotly} -> {dest_plotly}")
            else:
                shutil.copy2(src_plotly, dest_plotly)
                print(f"[deploy] copied plotly.min.js from {src_plotly}")
        else:
            print(f"WARNING: no plotly.min.js and none at {src_plotly}; "
                  f"the page will fall back to the CDN copy.")

    files = collect(results)
    if not files:
        sys.exit(f"ERROR: nothing matched {ALLOW_PATTERNS} under {results}")

    total = sum(f.stat().st_size for f in files)
    print(f"\n{len(files)} files, {total/1e6:.1f} MB total")
    for f in sorted(files, key=lambda p: -p.stat().st_size)[:10]:
        print(f"   {f.stat().st_size/1e6:9.2f} MB  {f.relative_to(results)}")
    if len(files) > 10:
        print(f"   ... and {len(files)-10} more")

    oversize = [f for f in files if f.stat().st_size > HF_MAX_FILE_BYTES]
    if oversize:
        sys.exit("ERROR: files exceed HuggingFace's 50 GB per-file limit:\n" +
                 "\n".join(f"  {f.relative_to(results)}" for f in oversize))
    big = [f for f in files if f.stat().st_size > WARN_FILE_BYTES]
    if big:
        print("\nWARNING: large shards -- a browser downloads one whole shard per "
              "subset click, so these will feel broken to a viewer:")
        for f in big:
            print(f"  {f.stat().st_size/1e6:.0f} MB  {f.relative_to(results)}")

    if args.dry_run:
        print(f"\n[dry-run] would deploy to https://huggingface.co/spaces/{args.repo_id}")
        print("[dry-run] nothing created, nothing uploaded.")
        return

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGINGFACE_TOKEN not set (expected in .env at the repo root).")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="space", space_sdk="static",
                private=args.private, exist_ok=True, token=token)
    print(f"[deploy] repo ready: {args.repo_id} (private={args.private})")

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(results),
        allow_patterns=ALLOW_PATTERNS,
        commit_message=args.commit_message,
    )
    print(f"[deploy] DONE -> https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
