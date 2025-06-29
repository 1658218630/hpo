
import math
import random
from typing import Dict, Any, List, Tuple, Optional
from threading import Lock

# Allow both flat and package import layouts
try:
    from BaseOptimizer import BaseOptimizer
except ModuleNotFoundError:  # project might use a package structure
    from optimizers.BaseOptimizer import BaseOptimizer


class SuccessiveHalvingOptimizer(BaseOptimizer):
    """Synchronous Successive‑Halving (SHA) implementation."""

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        hyperparameter_space: Dict[str, Any],
        *,
        eta: int = 3,
        budget_key: str = "epochs",
        min_budget: Optional[int] = None,
        max_budget: Optional[int] = None,
        name: str = "SuccessiveHalving",
    ) -> None:
        self.space = hyperparameter_space
        self.budget_key = budget_key
        self.eta = max(2, int(eta))
        self._rng = random.Random()
        self._lock = Lock()

        # ------------------------- derive budget range ----------------------
        hp_budget_spec = hyperparameter_space.get(budget_key)

        def _extract_min_max(spec):
            if spec is None:
                return None, None
            # (low, high)
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                return spec[0], spec[1]
            # dict with numeric values
            if isinstance(spec, dict):
                cand_vals = [v for v in spec.values() if isinstance(v, (int, float))]
                if not cand_vals:
                    return None, None
                return min(cand_vals), max(cand_vals)
            # single numeric interpreted as *max*
            if isinstance(spec, (int, float)):
                return None, spec
            return None, None

        derived_min, derived_max = _extract_min_max(hp_budget_spec)
        self.min_budget = int(min_budget if min_budget is not None else (derived_min if derived_min is not None else 1))

        if max_budget is not None:
            self.max_budget = int(max_budget)
        elif derived_max is not None:
            self.max_budget = int(derived_max)
        else:
            self.max_budget = int(self.min_budget * (self.eta ** 3))  # default 4 generations

        if self.max_budget < self.min_budget:
            raise ValueError(f"max_budget ({self.max_budget}) must be ≥ min_budget ({self.min_budget})")

        # how many generations fit between min and max budgets
        self.max_generations = int(math.floor(math.log(self.max_budget / self.min_budget, self.eta))) + 1

        # ------------------------- bookkeeping -----------------------------
        super().__init__(name=name, hyperparameter_count=len(hyperparameter_space))
        self._generation: int = 0
        self._current_budget: int = self.min_budget
        self._queue: List[Dict[str, Any]] = []
        self._pending: int = 0
        self._scores: List[Tuple[Dict[str, Any], float]] = []

        # population size of the first generation
        self._initial_population = self.eta ** self.max_generations
        self._populate_initial_queue()

    # ------------------------------------------------------------------ #
    # BaseOptimizer API
    # ------------------------------------------------------------------ #
    def suggest_hyperparameters(self, round_num: int) -> Dict[str, Any]:
        with self._lock:
            if not self._queue:
                self._refill_queue()

            # ensure queue non-empty; otherwise fallback to random @ max_budget
            if not self._queue:
                cfg = self._sample_random_config()
                cfg[self.budget_key] = self.max_budget
                return cfg

            cfg = self._queue.pop(0)
            self._pending += 1
            return cfg

    def update(self, hyperparameters: Dict[str, Any], score: float) -> None:
        with self._lock:
            self.history.append((hyperparameters, score))
            if score > self.best_score:
                self.best_score = score
                self.best_params = hyperparameters

            self._pending -= 1
            self._scores.append((hyperparameters, score))

            # generation finished?
            if self._pending == 0 and not self._queue:
                self._advance_generation()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sample_random_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for name, spec in self.space.items():
            if name == self.budget_key:
                continue
            if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    cfg[name] = self._rng.randint(int(low), int(high))
                else:
                    cfg[name] = self._rng.uniform(float(low), float(high))
            elif isinstance(spec, (list, tuple)):
                cfg[name] = self._rng.choice(spec)
            else:
                cfg[name] = spec
        return cfg

    def _populate_initial_queue(self) -> None:
        self._generation = 0
        self._current_budget = self.min_budget
        self._queue = [
            {**self._sample_random_config(), self.budget_key: self._current_budget}
            for _ in range(self._initial_population)
        ]
        self._rng.shuffle(self._queue)
        self._scores.clear()
        self._pending = 0

    def _refill_queue(self) -> None:
        if self._queue:
            return

        if self._generation < self.max_generations - 1:
            self._advance_generation()
        else:
            # finished all generations – keep exploring random configs
            cfg = self._sample_random_config()
            cfg[self.budget_key] = self.max_budget
            self._queue.append(cfg)

    def _advance_generation(self) -> None:
        if not self._scores:
            # nothing to promote (edge case)
            return

        # sort by score (higher better) and keep top 1/eta
        self._scores.sort(key=lambda t: t[1], reverse=True)
        survivors = [cfg for cfg, _ in self._scores[: max(1, len(self._scores) // self.eta)]]

        self._generation += 1
        self._current_budget = min(self._current_budget * self.eta, self.max_budget)

        self._queue = []
        for cfg in survivors:
            cfg = cfg.copy()
            cfg[self.budget_key] = self._current_budget
            self._queue.append(cfg)
        self._rng.shuffle(self._queue)

        self._scores.clear()
        self._pending = 0
