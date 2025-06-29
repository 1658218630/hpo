import numpy as np
import time
import pandas as pd
from threading import Lock
from typing import Dict, List


class TournamentResults:
    """Class to track and manage tournament results"""

    def __init__(self):
        self.results = {}
        self.lock = Lock()
        self.start_time = time.time()

    def add_result(
        self,
        optimizer: str,
        round_num: int,
        hyperparams: Dict,
        score: float,
        training_time: float,
    ):
        """Add a result to the tournament tracking"""
        with self.lock:
            if optimizer not in self.results:
                self.results[optimizer] = {
                    "rounds": [],
                    "scores": [],
                    "hyperparams": [],
                    "training_times": [],
                    "best_score": -np.inf,
                    "best_round": 0,
                    "best_hyperparams": None,
                }

            self.results[optimizer]["rounds"].append(round_num)
            self.results[optimizer]["scores"].append(score)
            self.results[optimizer]["hyperparams"].append(hyperparams.copy())
            self.results[optimizer]["training_times"].append(training_time)

            # Update best score
            if score > self.results[optimizer]["best_score"]:
                self.results[optimizer]["best_score"] = score
                self.results[optimizer]["best_round"] = round_num
                self.results[optimizer]["best_hyperparams"] = hyperparams.copy()

    def get_current_standings(self) -> pd.DataFrame:
        """Get current tournament standings as DataFrame"""
        standings = []
        for optimizer, result in self.results.items():
            if result["scores"]:  # Only include if there are results
                standings.append(
                    {
                        "Rank": 0,  # Will be set after sorting
                        "Optimizer": optimizer,
                        "Best Score": result["best_score"],
                        "Best Round": result["best_round"],
                        "Total Rounds": len(result["rounds"]),
                        "Avg Training Time": np.mean(result["training_times"]),
                    }
                )

        if not standings:
            return pd.DataFrame()

        df = pd.DataFrame(standings)
        df = df.sort_values("Best Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        return df


class MultiSeedTournamentResults:
    """Enhanced class to track tournament results across multiple seeds"""

    def __init__(self, seeds):
        self.seeds = seeds
        self.seed_results = {seed: {} for seed in seeds}
        self.aggregated_results = {}
        self.lock = Lock()
        self.start_time = time.time()

    def add_result(
        self,
        seed: int,
        optimizer: str,
        round_num: int,
        hyperparams: Dict,
        score: float,
        training_time: float,
    ):
        """Add a result for a specific seed"""
        with self.lock:
            if optimizer not in self.seed_results[seed]:
                self.seed_results[seed][optimizer] = {
                    "rounds": [],
                    "scores": [],
                    "hyperparams": [],
                    "training_times": [],
                    "best_score": -np.inf,
                    "best_round": 0,
                    "best_hyperparams": None,
                }

            result = self.seed_results[seed][optimizer]
            result["rounds"].append(round_num)
            result["scores"].append(score)
            result["hyperparams"].append(hyperparams.copy())
            result["training_times"].append(training_time)

            # Update best score for this seed
            if score > result["best_score"]:
                result["best_score"] = score
                result["best_round"] = round_num
                result["best_hyperparams"] = hyperparams.copy()

            # Update aggregated results
            self._update_aggregated_results()

    def _update_aggregated_results(self):
        """Update aggregated statistics across all seeds"""
        optimizers = set()
        for seed_data in self.seed_results.values():
            optimizers.update(seed_data.keys())

        for optimizer in optimizers:
            # Collect data across all seeds
            all_scores = []
            all_best_scores = []
            all_training_times = []

            for seed in self.seeds:
                if optimizer in self.seed_results[seed]:
                    seed_data = self.seed_results[seed][optimizer]
                    if seed_data["scores"]:
                        all_scores.extend(seed_data["scores"])
                        all_best_scores.append(seed_data["best_score"])
                        all_training_times.extend(seed_data["training_times"])

            if all_scores:
                self.aggregated_results[optimizer] = {
                    "mean_score": np.mean(all_scores),
                    "std_score": np.std(all_scores),
                    "mean_best_score": np.mean(all_best_scores),
                    "std_best_score": np.std(all_best_scores),
                    "mean_training_time": np.mean(all_training_times),
                    "std_training_time": np.std(all_training_times),
                    "best_scores_per_seed": all_best_scores,
                    "total_evaluations": len(all_scores),
                    "seeds_completed": len(
                        [
                            s
                            for s in self.seeds
                            if optimizer in self.seed_results[s]
                            and self.seed_results[s][optimizer]["scores"]
                        ]
                    ),
                }

    def get_current_standings_with_variance(self) -> pd.DataFrame:
        """Get current tournament standings with variance information"""
        standings = []
        for optimizer, stats in self.aggregated_results.items():
            if stats["total_evaluations"] > 0:
                standings.append(
                    {
                        "Rank": 0,
                        "Optimizer": optimizer,
                        "Mean Best Score": stats["mean_best_score"],
                        "Std Best Score": stats["std_best_score"],
                        "Seeds Completed": stats["seeds_completed"],
                        "Total Seeds": len(self.seeds),
                        "Total Evaluations": stats["total_evaluations"],
                        "Mean Training Time": stats["mean_training_time"],
                        "Std Training Time": stats["std_training_time"],
                    }
                )

        if not standings:
            return pd.DataFrame()

        df = pd.DataFrame(standings)
        df = df.sort_values("Mean Best Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        return df

    def get_score_history_by_seed(self, optimizer: str) -> Dict[int, List[float]]:
        """Get score history for an optimizer across all seeds"""
        history = {}
        for seed in self.seeds:
            if optimizer in self.seed_results[seed]:
                history[seed] = self.seed_results[seed][optimizer]["scores"]
            else:
                history[seed] = []
        return history
