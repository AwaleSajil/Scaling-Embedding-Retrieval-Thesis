"""
Statistical significance testing between model pairs.

For each (model_a, model_b, subset, metric) combination, computes:
  - p-value via Wilcoxon signed-rank test (two-sided, paired per query)
  - effect size via Cohen's d
  - wins / ties / losses counts

Results are stored as a flat dict keyed by "{model_a}|||{model_b}|||{subset}|||{metric}".

The special subset "mean" pools all per-query scores from every real subset
(i.e. queries across all subsets are concatenated) so significance can be
assessed over the full evaluation set.
"""
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from eval_v2.results.store import ResultsStore


def _cohens_d(a: list[float], b: list[float]) -> float:
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def _win_tie_loss(a: list[float], b: list[float]) -> tuple[int, int, int]:
    wins = sum(x > y for x, y in zip(a, b))
    ties = sum(x == y for x, y in zip(a, b))
    losses = sum(x < y for x, y in zip(a, b))
    return wins, ties, losses


def compute_significance(
    store: "ResultsStore",
    min_queries: int = 5,
    existing: dict | None = None,
) -> dict[str, dict]:
    """
    Compute pairwise significance for all
    (model_a, model_b) × subset × metric combinations.

    Keys already present in `existing` are skipped. Pass an empty dict (or omit)
    to compute everything from scratch.

    Returns only the newly computed entries (caller merges with existing).
    """
    existing = existing or {}
    try:
        from scipy.stats import wilcoxon, ttest_rel, shapiro
    except ImportError:
        warnings.warn("scipy not installed — significance testing skipped.")
        return {}

    per_query = store.load_per_query()
    model_keys = list(per_query.keys())
    results: dict[str, dict] = {}

    # Collect all subsets and all metrics across all models
    all_subsets: set[str] = set()
    all_metrics: set[str] = set()
    for model_data in per_query.values():
        for subset, queries in model_data.items():
            all_subsets.add(subset)
            for qdata in queries.values():
                all_metrics.update(
                    k for k in qdata
                    if k not in ("text", "relevant_ids", "ranked_ids", "ranked_scores")
                )

    real_subsets = sorted(s for s in all_subsets if s != "mean")

    def _run_significance(subset_label: str, model_score_arrays: dict[str, list[float]], n: int) -> None:
        """Compute and store pairwise significance for a pre-built set of score arrays."""
        for i, model_a in enumerate(model_keys):
            for model_b in model_keys[i + 1:]:
                key = f"{model_a}|||{model_b}|||{subset_label}|||{metric}"
                if key in existing:
                    continue
                scores_a = model_score_arrays.get(model_a, [])
                scores_b = model_score_arrays.get(model_b, [])
                if not scores_a or not scores_b:
                    continue
                if any(np.isnan(x) for x in scores_a + scores_b):
                    continue

                diff = np.array(scores_a) - np.array(scores_b)
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(scores_a, scores_b, alternative="two-sided")
                    except Exception:
                        p_val = float("nan")

                try:
                    t_stat_val, t_p_val = ttest_rel(scores_a, scores_b)
                except Exception:
                    t_stat_val, t_p_val = float("nan"), float("nan")

                try:
                    sw_stat_val, sw_p_val = (
                        shapiro(diff) if len(diff) >= 3 else (float("nan"), float("nan"))
                    )
                except Exception:
                    sw_stat_val, sw_p_val = float("nan"), float("nan")

                results[key] = {
                    "p": float(p_val),
                    "t_stat": float(t_stat_val),
                    "t_p": float(t_p_val),
                    "sw_stat": float(sw_stat_val),
                    "sw_p": float(sw_p_val),
                    "effect": _cohens_d(scores_a, scores_b),
                    **dict(zip(("wins", "ties", "losses"), _win_tie_loss(scores_a, scores_b))),
                    "n": n,
                }

    # ── Per-subset significance ────────────────────────────────────────
    for subset in real_subsets:
        common_qids = None
        for model_key in model_keys:
            qids = set(per_query.get(model_key, {}).get(subset, {}).keys())
            common_qids = qids if common_qids is None else common_qids & qids
        if not common_qids or len(common_qids) < min_queries:
            continue
        common_qids = sorted(common_qids)

        for metric in sorted(all_metrics):
            model_scores: dict[str, list[float]] = {}
            for model_key in model_keys:
                subset_data = per_query.get(model_key, {}).get(subset, {})
                scores = [subset_data.get(qid, {}).get(metric, float("nan")) for qid in common_qids]
                model_scores[model_key] = scores
            _run_significance(subset, model_scores, len(common_qids))

    # ── Pooled "mean" significance ─────────────────────────────────────
    # Concatenate per-query scores from all real subsets using queries that
    # are present in every model for that subset, so the pairing is preserved.
    for metric in sorted(all_metrics):
        pooled: dict[str, list[float]] = {k: [] for k in model_keys}
        for subset in real_subsets:
            common_qids = None
            for model_key in model_keys:
                qids = set(per_query.get(model_key, {}).get(subset, {}).keys())
                common_qids = qids if common_qids is None else common_qids & qids
            if not common_qids or len(common_qids) < min_queries:
                continue
            common_qids = sorted(common_qids)
            for model_key in model_keys:
                subset_data = per_query.get(model_key, {}).get(subset, {})
                scores = [subset_data.get(qid, {}).get(metric, float("nan")) for qid in common_qids]
                if any(np.isnan(s) for s in scores):
                    continue
                pooled[model_key].extend(scores)

        n_pooled = len(next(iter(pooled.values()), []))
        if n_pooled < min_queries:
            continue
        _run_significance("mean", pooled, n_pooled)

    return results
