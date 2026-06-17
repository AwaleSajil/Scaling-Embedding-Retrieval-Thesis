"""Identify local W&B runs: backbone, experiment, group, display name, epochs, gamma."""
import json, glob, os, yaml

EXP = {
    "e1": ("Baseline", "FT"), "e2": ("BAT", "BAT-STE"), "e3": ("MRL", "MRL"),
    "e4": ("MRL+BAT", "MRL+BAT"), "e5": ("AnnealedTanh", "AnnTanh"),
    "e7": ("MRL+AnnTanh", "MRL+AnnTanh"),
    "e6_1b": ("MultiBitASigm", "ASigm-1b"), "e6_2b": ("MultiBitASigm", "ASigm-2b"),
    "e6_3b": ("MultiBitASigm", "ASigm-3b"), "e6_4b": ("MultiBitASigm", "ASigm-4b"),
}
BB = {"BAAI/bge-base-en-v1.5": "bge", "FacebookAI/roberta-base": "roberta",
      "microsoft/mpnet-base": "mpnet"}


def get(cfg, *path, default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
        if isinstance(cur, dict) and "value" in cur and set(cur.keys()) <= {"value", "desc"}:
            cur = cur["value"]
    return cur


rows = []
for d in sorted(glob.glob("wandb/run-*")):
    rid = d.split("-")[-1]
    cfgp, metap = f"{d}/files/config.yaml", f"{d}/files/wandb-metadata.json"
    exp = bb = gamma = epochs = ts = None
    try:
        cfg = yaml.safe_load(open(cfgp))
        exp = get(cfg, "experiment", "number")
        bb = get(cfg, "input_model", "name")
        gamma = get(cfg, "annealed_tanh_config", "gamma")
        epochs = get(cfg, "num_train_epochs") or get(cfg, "trainer_config", "num_train_epochs")
        od = get(cfg, "output_dir") or ""
        ts = od.split("timestamp_")[-1].split("/")[0] if "timestamp_" in od else ""
    except Exception:
        pass
    if exp is None and os.path.exists(metap):
        try:
            a = json.load(open(metap)).get("args", [])
            if "--expirement_number" in a:
                exp = a[a.index("--expirement_number") + 1]
            if "--model_name" in a:
                bb = a[a.index("--model_name") + 1]
        except Exception:
            pass
    grp, disp = EXP.get(exp, ("?", "?"))
    rows.append((rid, BB.get(bb, bb or "?"), exp or "?", grp, disp,
                 str(epochs or ""), str(gamma or ""), ts or ""))

hdr = f"{'run_id':<12}{'backbone':<9}{'exp':<6}{'group':<14}{'display':<12}{'ep':<3}{'gamma':<6}{'timestamp'}"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x[1], x[2], x[7])):
    print(f"{r[0]:<12}{r[1]:<9}{r[2]:<6}{r[3]:<14}{r[4]:<12}{r[5]:<3}{r[6]:<6}{r[7]}")
print(f"\n{len(rows)} runs")
