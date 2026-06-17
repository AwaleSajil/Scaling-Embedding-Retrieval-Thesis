"""Tag (and optionally rename) synced W&B runs by reading each run's own config.

Dry-run by default — prints planned changes without modifying anything.
Apply with --apply.  Rename runs too with --rename.

  python tag_runs.py                 # preview
  python tag_runs.py --apply         # set tags
  python tag_runs.py --apply --rename
"""
import argparse
import re
import sys
from collections import defaultdict

import wandb

sys.path.insert(0, "/rhome/sawale/thesis")
from eval_v2.config.models import MODELS

# Index ModelSpec paths so a run can be matched to its registry key(s).
_PATH_KEYS = defaultdict(list)       # exact spec.path -> [keys]
_TSDIR_KEYS = defaultdict(list)      # "timestamp_<ts>/<backbone>" -> [keys]
for _k, _spec in MODELS.items():
    _PATH_KEYS[_spec.path].append(_k)
    _m = re.search(r"(timestamp_[0-9_-]+/[^/]+)", _spec.path)
    if _m:
        _TSDIR_KEYS[_m.group(1)].append(_k)


def model_key_for_run(cfg):
    """Return the representative MODELS key for a run, matched by its output path."""
    od = (_val(cfg, "output_dir") or "").rstrip("/")
    if not od:
        return None
    cands = _PATH_KEYS.get(od + "/final_model", [])
    if not cands:
        m = re.search(r"(timestamp_[0-9_-]+/[^/]+)", od)
        if m:
            cands = _TSDIR_KEYS.get(m.group(1), [])
    if not cands:
        return None
    return min(cands, key=len)  # shortest key == the base training variant


EXP = {
    "e1": ("Baseline", "FT"), "e2": ("BAT", "BAT-STE"), "e3": ("MRL", "MRL"),
    "e4": ("MRL+BAT", "MRL+BAT"), "e5": ("AnnealedTanh", "AnnTanh"),
    "e7": ("MRL+AnnTanh", "MRL+AnnTanh"),
    "e6_1b": ("MultiBitASigm", "ASigm-1b"), "e6_2b": ("MultiBitASigm", "ASigm-2b"),
    "e6_3b": ("MultiBitASigm", "ASigm-3b"), "e6_4b": ("MultiBitASigm", "ASigm-4b"),
}
BB = {"BAAI/bge-base-en-v1.5": "bge", "FacebookAI/roberta-base": "roberta",
      "microsoft/mpnet-base": "mpnet"}


def _val(cfg, key, sub=None, default=None):
    """Read a possibly-nested config value, tolerating wandb's {value,desc} wrap."""
    v = cfg.get(key, default)
    if isinstance(v, dict) and "value" in v and set(v.keys()) <= {"value", "desc"}:
        v = v["value"]
    if sub is not None:
        v = v.get(sub, default) if isinstance(v, dict) else default
    return v


def derive(cfg):
    exp = _val(cfg, "experiment", "number")
    bb = _val(cfg, "input_model", "name")
    gamma = _val(cfg, "annealed_tanh_config", "gamma")
    epochs = _val(cfg, "num_train_epochs") or _val(cfg, "trainer_config", "num_train_epochs")
    grp, disp = EXP.get(exp, (None, None))
    bb_short = BB.get(bb, bb)
    tags = []
    if bb_short: tags.append(f"backbone:{bb_short}")
    if exp:      tags.append(f"exp:{exp}")
    if grp:      tags.append(f"group:{grp}")
    if disp:     tags.append(f"method:{disp}")
    if epochs:   tags.append(f"epochs:{epochs}")
    if exp == "e5" and gamma is not None: tags.append(f"gamma:{gamma}")
    # Run name = its MODELS registry key (matched by output path); fall back to
    # a constructed name if the run's checkpoint isn't in the registry.
    name = model_key_for_run(cfg)
    # A trajectory training run produced the whole -nstep-* family, not one step;
    # name it by the base variant + epoch count (unique vs the 1-epoch key).
    if name and "-nstep-" in name:
        base = re.sub(r"-nstep-\d+$", "", name)
        name = f"{base}-{epochs}ep" if epochs else base
    if not name and bb_short and disp:
        name = f"{bb_short}-{disp}" + (f"-{epochs}ep" if epochs else "")
        if exp == "e5" and gamma is not None:
            name += f"-g{gamma}"
    if name:
        tags.append(f"key:{name}")
    return tags, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="impact-ibm-collaboration")
    ap.add_argument("--project", default="scale_emb_retrieval")
    ap.add_argument("--apply", action="store_true", help="actually write changes")
    ap.add_argument("--rename", action="store_true", help="also set a friendly run name")
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")
    print(f"{'run_id':<16}{'name->':<26}{'tags'}")
    print("-" * 90)
    n = 0
    for run in runs:
        tags, name = derive(run.config)
        if not tags:
            print(f"{run.id:<16}{'(no config — skipped)':<26}")
            continue
        newname = (name if args.rename and name else run.name)
        print(f"{run.id:<16}{newname:<26}{','.join(tags)}")
        if args.apply:
            run.tags = sorted(set(run.tags) | set(tags))
            if args.rename and name:
                run.name = name
            run.update()
            n += 1
    print(f"\n{'APPLIED to' if args.apply else 'WOULD tag'} {n if args.apply else 'N'} runs"
          f"{' (dry-run; pass --apply)' if not args.apply else ''}")


if __name__ == "__main__":
    main()
