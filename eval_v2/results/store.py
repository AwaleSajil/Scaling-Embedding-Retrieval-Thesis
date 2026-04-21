"""
Persistent result storage for eval_v2.

Files written under <output_dir>/:
    aggregate.json     — model → subset → metric@k → float
    per_query.json     — model → subset → query_id → {metrics, ranked_ids, relevant_ids}
    emb_info.json      — model → subset → {queries: {...}, corpus: {...}}
    significance.json  — key → {p, effect, wins, ties, losses, n}

All files are updated incrementally so partial runs can be resumed.
"""
import json
import os
from pathlib import Path
from typing import Any

from eval_v2.core.evaluator import SubsetResult


class ResultsStore:
    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self._agg_path = self.out / "aggregate.json"
        self._pq_path = self.out / "per_query.json"
        self._emb_path = self.out / "emb_info.json"
        self._sig_path = self.out / "significance.json"

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_subset_result(self, model_key: str, result: SubsetResult) -> None:
        """Persist aggregate metrics, per-query data, and embedding info for one result."""
        subset = result.subset

        # --- aggregate ---
        agg = self._load(self._agg_path)
        agg.setdefault(model_key, {})[subset] = result.aggregate
        self._save(self._agg_path, agg)

        # --- per_query ---
        pq = self._load(self._pq_path)
        pq.setdefault(model_key, {})[subset] = {
            qr.query_id: {
                "text": qr.query_text,
                "relevant_ids": list(qr.relevant_ids),
                "ranked_ids": qr.ranked_ids,
                "ranked_scores": qr.ranked_scores,
                **qr.metrics,
            }
            for qr in result.per_query
        }
        self._save(self._pq_path, pq)

        # --- emb_info ---
        emb = self._load(self._emb_path)
        emb.setdefault(model_key, {})[subset] = result.emb_info
        self._save(self._emb_path, emb)

    def save_mean_metrics(self, model_key: str, subsets: list[str]) -> None:
        """Compute and store mean aggregate metrics across *subsets* for *model_key*."""
        agg = self._load(self._agg_path)
        model_data = agg.get(model_key, {})

        per_metric: dict[str, list[float]] = {}
        for subset in subsets:
            for metric, val in model_data.get(subset, {}).items():
                per_metric.setdefault(metric, []).append(val)

        mean_metrics = {m: sum(vs) / len(vs) for m, vs in per_metric.items() if vs}
        agg.setdefault(model_key, {})["mean"] = mean_metrics
        self._save(self._agg_path, agg)

        # Also aggregate emb_info across subsets
        emb = self._load(self._emb_path)
        model_emb = emb.get(model_key, {})
        agg_emb: dict[str, Any] = {"queries": {"n": 0, "dimension": 0, "element_size_bit": 0},
                                    "corpus": {"n": 0, "dimension": 0, "element_size_bit": 0}}
        for subset in subsets:
            info = model_emb.get(subset, {})
            for split in ("queries", "corpus"):
                agg_emb[split]["n"] += info.get(split, {}).get("n", 0)
                agg_emb[split]["dimension"] = info.get(split, {}).get("dimension", 0)
                agg_emb[split]["element_size_bit"] = info.get(split, {}).get("element_size_bit", 0)
        emb.setdefault(model_key, {})["mean"] = agg_emb
        self._save(self._emb_path, emb)

    def save_significance(self, sig: dict) -> None:
        self._save(self._sig_path, sig)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_aggregate(self) -> dict:
        return self._load(self._agg_path)

    def load_per_query(self) -> dict:
        return self._load(self._pq_path)

    def load_emb_info(self) -> dict:
        return self._load(self._emb_path)

    def load_significance(self) -> dict:
        return self._load(self._sig_path)

    def already_evaluated(self, model_key: str, subset: str) -> bool:
        agg = self._load(self._agg_path)
        return subset in agg.get(model_key, {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
