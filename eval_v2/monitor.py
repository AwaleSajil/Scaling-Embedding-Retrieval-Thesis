"""
Weights & Biases progress reporting for eval_v2 runs.

Why this exists: a full BEIR run is multi-day, and the only way to see how far
along it is otherwise is to be logged into the cluster reading SLURM logs. W&B
gives a URL that works from anywhere, and -- unlike a self-hosted status page --
it detects a dead run on its own: if the job crashes, the run stops heartbeating
and W&B marks it `crashed` without anyone having to notice a stale timestamp.

Design notes:
  * Entirely optional. Without --wandb every method is a no-op, so the eval
    behaves exactly as before.
  * Never fatal. A monitoring failure must not kill a multi-day evaluation, so
    every W&B call is wrapped; problems degrade to a printed warning.
  * Runs are named after the SLURM job (`beir-full-3924500`) so a run in the UI
    maps directly onto slurm_logs/3924500_%x.out and `sacct -j 3924500`.
    Training runs use W&B's auto-generated names (`denim-morning-3`), which
    cannot be traced back to anything -- hence the explicit scheme here.
"""
from __future__ import annotations

import os
import time


class NullMonitor:
    """No-op stand-in used when monitoring is disabled."""

    enabled = False

    def subset_start(self, *a, **k): pass
    def pair_done(self, *a, **k): pass
    def stage(self, *a, **k): pass
    def finish(self, *a, **k): pass

    @property
    def url(self) -> str | None:
        return None


class WandbMonitor(NullMonitor):
    enabled = True

    def __init__(self, project: str, name: str | None, tags: list[str],
                 config: dict, job_type: str = "eval"):
        import wandb

        self._wandb = wandb
        self._t0 = time.time()
        self._pairs = 0
        self._expected = config.get("n_models", 0) * config.get("n_subsets", 0)
        self._run = None

        key = os.environ.get("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
        self._run = wandb.init(
            project=project,
            name=name,
            job_type=job_type,
            tags=tags,
            config=config,
            # Under SLURM stdout is redirected to a file, and W&B's default
            # console handling captures nothing -- the run ends up with no
            # output.log and an empty Logs tab. "wrap" forces capture.
            settings=wandb.Settings(console="wrap"),
        )
        print(f"[wandb] {self.url}")
        self._attach_slurm_logs()

    @property
    def url(self) -> str | None:
        try:
            return self._run.url if self._run else None
        except Exception:
            return None

    def _safe(self, fn, *a, **k):
        try:
            fn(*a, **k)
        except Exception as e:  # monitoring must never abort the eval
            print(f"[wandb] WARNING: {type(e).__name__}: {str(e)[:120]}")

    def _attach_slurm_logs(self):
        """Mirror the SLURM .out/.err into the run's Files tab.

        Console capture only sees output produced after wandb.init(), which
        misses everything the shell wrapper printed first (env selection, GPU
        check, model list) and all of stderr's progress bars. The SLURM files
        have the complete picture, so attach them with policy="live" to keep
        them syncing as the job writes.
        """
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            return
        from pathlib import Path
        base = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")) / "slurm_logs"
        found = sorted(base.glob(f"{job}_*.out")) + sorted(base.glob(f"{job}_*.err"))
        for p in found:
            # base_path keeps them at the root of the run's file tree
            self._safe(self._wandb.save, str(p), base_path=str(base), policy="live")
        if found:
            print(f"[wandb] mirroring {len(found)} SLURM log file(s)")

    def subset_start(self, subset: str, n_corpus: int, n_queries: int):
        self._safe(self._wandb.log, {
            "subset/name": subset,
            "subset/corpus_size": n_corpus,
            "subset/n_queries": n_queries,
            "progress/pairs_done": self._pairs,
        })

    def pair_done(self, model_key: str, subset: str, aggregate: dict):
        """Called after each (model, subset) result is stored."""
        self._pairs += 1
        payload = {
            "progress/pairs_done": self._pairs,
            "progress/pairs_total": self._expected,
            "progress/fraction": self._pairs / self._expected if self._expected else 0.0,
            "progress/elapsed_hours": (time.time() - self._t0) / 3600,
        }
        # Headline metric per subset, so the UI shows quality trends too, not
        # just a counter. Keyed by subset to keep series separate.
        for m in ("ndcg@10", "recall@10", "mrr@10"):
            if m in aggregate:
                payload[f"{subset}/{m}"] = aggregate[m]
        self._safe(self._wandb.log, payload)

    def stage(self, name: str, **fields):
        """Mark a coarse pipeline stage (significance, html, ...)."""
        self._safe(self._wandb.log, {f"stage/{name}": 1, **fields})

    def finish(self, ok: bool = True, note: str = ""):
        def _end():
            self._wandb.summary["pairs_done"] = self._pairs
            self._wandb.summary["elapsed_hours"] = (time.time() - self._t0) / 3600
            try:
                self._wandb.alert(
                    title="BEIR eval finished" if ok else "BEIR eval failed",
                    text=note or f"{self._pairs}/{self._expected} pairs in "
                                 f"{(time.time()-self._t0)/3600:.1f} h",
                )
            except Exception:
                pass  # alerts are unavailable on some plans; not worth failing
            self._wandb.finish(exit_code=0 if ok else 1)
        self._safe(_end)


def make_monitor(args, models: dict, subsets: list[str]) -> NullMonitor:
    """Build a monitor from parsed CLI args. Returns NullMonitor when disabled."""
    if not getattr(args, "wandb", False):
        return NullMonitor()
    if os.environ.get("WANDB_MODE") == "disabled":
        print("[wandb] WANDB_MODE=disabled; monitoring off")
        return NullMonitor()

    job = os.environ.get("SLURM_JOB_ID", "local")
    name = args.wandb_name or f"{args.dataset}-{args.wandb_run_kind}-{job}"
    tags = [args.dataset, "eval", args.wandb_run_kind]
    n_ckpt = len({s.path for s in models.values()})
    config = {
        "dataset": args.dataset,
        "subsets": subsets,
        "n_subsets": len(subsets),
        "n_models": len(models),
        "n_checkpoints": n_ckpt,
        "batch_size": args.batch_size,
        "slurm_job_id": job,
        "ks": args.ks,
    }
    try:
        return WandbMonitor(args.wandb_project, name, tags, config)
    except Exception as e:
        print(f"[wandb] WARNING: could not start monitoring "
              f"({type(e).__name__}: {str(e)[:120]}); continuing without it.")
        return NullMonitor()
