"""Simple query statistics tracker used during attack execution."""

from __future__ import annotations

from dataclasses import dataclass

from mcg.domain.interfaces import IQueryStats, QuerySummary
from utils.attack_count import AttackCountingFunction


@dataclass
class QueryTracker(IQueryStats):
    """Thin adapter around ``AttackCountingFunction`` used in the legacy pipeline."""

    _tracker: AttackCountingFunction

    def update(self, queries: int, success: bool) -> None:
        self._tracker.add(int(queries), bool(success))

    def summary(self) -> QuerySummary:
        total_attempts = len(self._tracker.all_counts)
        if total_attempts == 0:
            return QuerySummary(
                mean_queries=0.0,
                median_queries=0.0,
                first_success_rate=0.0,
                success_rate=0.0,
                total_attempts=0,
            )

        mean_queries = float(self._tracker.get_average())
        median_queries = float(self._tracker.get_median())
        first_success_rate = float(self._tracker.get_first_success())
        success_rate = float(self._tracker.get_success_rate())
        return QuerySummary(
            mean_queries=mean_queries,
            median_queries=median_queries,
            first_success_rate=first_success_rate,
            success_rate=success_rate,
            total_attempts=total_attempts,
        )
