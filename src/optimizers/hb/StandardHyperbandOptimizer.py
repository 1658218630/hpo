import math
import random
import numpy as np
from typing import Dict, Any, List, Optional

from optimizers.BaseOptimizer import BaseOptimizer


class StandardHyperbandOptimizer(BaseOptimizer):
    """Synchronous **Successive Halving / Hyperband** scheduler compatible
    with tournament.ipynb; accepts only `search_space` as positional arg."""

    class _Trial:
        def __init__(self, tid: int, params: Dict[str, Any], budget: int,
                     bracket: int, rung: int) -> None:
            self.id = tid
            self.params = params
            self.budget = budget
            self.bracket = bracket
            self.rung = rung
            self.score: Optional[float] = None

    def __init__(self,
                 search_space: Dict[str, Any],
                 *,
                 name: str = "Hyperband",
                 eta: int = 3,
                 min_budget: int = 1,
                 max_budget: int = 81,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        search_space : dict
            Defines sampling for each hyperparameter.
        eta : int
            Reduction factor (≥2).
        min_budget : int
            Smallest resource (epochs).
        max_budget : int
            Largest resource (epochs).
        seed : int, optional
            RNG seed.
        """
        # Seed control
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Param checks
        if eta < 2:
            raise ValueError("eta must be ≥ 2")
        if max_budget <= min_budget:
            raise ValueError("max_budget must exceed min_budget")

        self.search_space = search_space
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # Infer count and init BaseOptimizer
        hp_count = len(search_space)
        super().__init__(name=name, hyperparameter_count=hp_count)

        # Hyperband specs
        self.s_max = int(math.floor(math.log(max_budget / min_budget, eta)))
        self.B = (self.s_max + 1) * max_budget
        self._brackets_spec: List[Any] = []
        for s in reversed(range(self.s_max + 1)):
            n0 = int(math.ceil(self.B / max_budget * (eta ** -s) / (s + 1)))
            r0 = int(max_budget * (eta ** -s))
            n_list = [max(1, int(n0 * (eta ** -i))) for i in range(s + 1)]
            r_list = [int(r0 * (eta ** i)) for i in range(s + 1)]
            self._brackets_spec.append((s, n_list, r_list))

        # State
        self._bracket_cursor = -1
        self._current_bracket: Optional[Dict[str, Any]] = None
        self._pending_queue: List[StandardHyperbandOptimizer._Trial] = []
        self._live_trials: Dict[int, StandardHyperbandOptimizer._Trial] = {}
        self._trial_id_counter = 0

    def suggest_hyperparameters(self, round_num: int) -> Dict[str, Any]:
        """Return next trial's hyperparams, budget, and id."""
        while not self._pending_queue:        
            self._maybe_generate_next_trials()


        trial = self._pending_queue.pop(0)
        self._live_trials[trial.id] = trial
        out = dict(trial.params)
        out["_budget"] = trial.budget
        out["_id"] = trial.id
        return out

    def update(self, hyperparameters: Dict[str, Any], score: float) -> None:
        """Ingest trial result, update internal state and best score."""
        trial_id = hyperparameters.get("_id")
        if trial_id is None or trial_id not in self._live_trials:
            raise ValueError("update() received unknown or already‑processed trial")
        trial = self._live_trials.pop(trial_id)
        trial.score = score
        # Record
        self.history.append((trial.params, trial.budget, score))
        # Best
        if score > self.best_score:
            self.best_score = score
            self.best_params = trial.params
        # Place into bracket
        bracket = self._current_bracket
        rung_state = bracket["rungs"][trial.rung]
        rung_state["results"].append(trial)
        # If rung complete, promote
        if len(rung_state["results"]) == rung_state["n"]:
            self._promote_survivors(bracket, trial.rung)

    def _maybe_generate_next_trials(self) -> None:
        # If bracket active, skip
        if self._current_bracket and self._current_bracket.get("active", False):
            return
        # Advance bracket
        self._bracket_cursor += 1
        if self._bracket_cursor >= len(self._brackets_spec):
            self._bracket_cursor = 0
        s, n_list, r_list = self._brackets_spec[self._bracket_cursor]
        rungs = [{"n": n, "budget": r, "results": []}
                 for n, r in zip(n_list, r_list)]
        self._current_bracket = {"s": s, "rungs": rungs, "active": True}
        # Seed first rung
        for _ in range(n_list[0]):
            self._pending_queue.append(
                self._create_trial(r_list[0], self._bracket_cursor, 0)
            )

    def _create_trial(self, budget: int, bracket: int, rung: int) -> _Trial:
        params = self._sample_config()
        t = StandardHyperbandOptimizer._Trial(
            tid=self._trial_id_counter,
            params=params,
            budget=budget,
            bracket=bracket,
            rung=rung,
        )
        self._trial_id_counter += 1
        return t

    def _promote_survivors(self, bracket_state: Dict[str, Any], rung_idx: int) -> None:
        rungs = bracket_state["rungs"]
        sorted_trials = sorted(rungs[rung_idx]["results"],
                               key=lambda t: t.score, reverse=True)
        survivors = sorted_trials[:max(1, rungs[rung_idx]["n"] // self.eta)]
        # End bracket
        if rung_idx + 1 >= len(rungs):
            bracket_state["active"] = False
            return
        # Schedule next rung
        next_rung = rungs[rung_idx + 1]
        for t in survivors:
            self._pending_queue.append(
                StandardHyperbandOptimizer._Trial(
                    tid=self._trial_id_counter,
                    params=t.params,
                    budget=next_rung["budget"],
                    bracket=t.bracket,
                    rung=rung_idx + 1,
                )
            )
            self._trial_id_counter += 1

    
    def _sample_config(self) -> Dict[str, Any]:
        """
        支持四种 search_space 描述：
        1) callable -> 直接调用
        2) list     -> 随机 choice
        3) tuple(lo, hi) -> 随机 int/float
        4) dict     -> {'choices': [...]} 或 {'min':…, 'max':…}
        5) 其它     -> 当常量
        """
        cfg: Dict[str, Any] = {}
        for k, spec in self.search_space.items():
            if callable(spec):
                cfg[k] = spec()

            elif isinstance(spec, list):
                cfg[k] = random.choice(spec)

            elif isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    cfg[k] = random.randint(lo, hi)
                else:
                    cfg[k] = random.uniform(float(lo), float(hi))

            elif isinstance(spec, dict):
                # 离散候选
                if 'choices' in spec and isinstance(spec['choices'], list):
                    cfg[k] = random.choice(spec['choices'])
                # 连续区间
                elif 'min' in spec and 'max' in spec:
                    lo, hi = spec['min'], spec['max']
                    if isinstance(lo, int) and isinstance(hi, int):
                        cfg[k] = random.randint(lo, hi)
                    else:
                        cfg[k] = random.uniform(lo, hi)
                else:
                    raise ValueError(f"Unsupported dict spec for '{k}': {spec}")

            else:
                # 常量／默认值
                cfg[k] = spec

        return cfg




