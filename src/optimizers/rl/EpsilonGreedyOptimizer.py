import numpy as np
import random
from typing import Dict, List
from optimizers.BaseOptimizer import BaseOptimizer


class EpsilonGreedyOptimizer(BaseOptimizer):
    """
    Epsilon-Greedy Multi-Armed Bandit for Hyperparameter Optimization

    This version creates new arms dynamically and learns which parameter
    regions work best.
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        epsilon: float = 0.5,
        region_splits: int = 5,
        epsilon_decay: float = 0.99,
    ):
        """
        Initialize Epsilon-Greedy Bandit with dynamic parameter regions

        Args:
            hyperparameter_space: Dictionary defining the hyperparameter search space
            epsilon: Exploration probability (0.5 = 50% exploration)
            region_splits: Number of regions to split each parameter into
            epsilon_decay: Decay factor for epsilon over time
        """
        super().__init__("EpsilonGreedy", region_splits ** len(hyperparameter_space))

        self.hyperparameter_space = hyperparameter_space
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.region_splits = region_splits

        # Create parameter regions for each hyperparameter
        self.parameter_regions = self._create_parameter_regions()

        # Track performance of each region combination
        self.region_rewards = {}  # Key: region_combo_tuple, Value: list of rewards
        self.region_counts = {}  # Key: region_combo_tuple, Value: count

        # Current round tracking
        self.current_round = 0
        self.last_suggested_regions = None

        print(
            f"EpsilonGreedy: epsilon={epsilon}, {len(self.parameter_regions)} parameters"
        )
        print(
            f"Parameter regions: {[(name, len(regions)) for name, regions in self.parameter_regions.items()]}"
        )

    def _create_parameter_regions(self) -> Dict[str, List]:
        """
        Create regions for each parameter to learn which regions work best

        Returns:
            Dictionary mapping parameter names to lists of region boundaries
        """
        regions = {}

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                # Create integer regions
                min_val, max_val = param_config["min"], param_config["max"]
                step = max(1, (max_val - min_val) // self.region_splits)
                param_regions = []
                for i in range(self.region_splits):
                    region_min = min_val + i * step
                    region_max = min(max_val, min_val + (i + 1) * step)
                    param_regions.append((region_min, region_max))
                regions[param_name] = param_regions

            elif param_config["type"] == "float":
                # Create float regions
                min_val, max_val = param_config["min"], param_config["max"]
                step = (max_val - min_val) / self.region_splits
                param_regions = []
                for i in range(self.region_splits):
                    region_min = min_val + i * step
                    region_max = min_val + (i + 1) * step
                    param_regions.append((region_min, region_max))
                regions[param_name] = param_regions

            elif param_config["type"] == "categorical":
                # Each categorical value is its own "region"
                regions[param_name] = [(val,) for val in param_config["values"]]

        return regions

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """
        Suggest hyperparameters using epsilon-greedy region selection
        """
        self.current_round = round_num

        # Decay epsilon over time
        current_epsilon = max(
            0.05, self.initial_epsilon * (self.epsilon_decay**round_num)
        )

        # Choose regions for each parameter
        selected_regions = {}

        if random.random() < current_epsilon or len(self.region_rewards) == 0:
            # EXPLORATION: Random region selection
            for param_name, param_regions in self.parameter_regions.items():
                selected_regions[param_name] = random.choice(range(len(param_regions)))
            decision_type = "EXPLORATION"
            print(f"Round {round_num}: {decision_type} (epsilon={current_epsilon:.3f})")
        else:
            # EXPLOITATION: Choose best known region combination
            selected_regions = self._get_best_region_combination()
            decision_type = "EXPLOITATION"
            best_combo = tuple(
                selected_regions[param] for param in sorted(selected_regions.keys())
            )
            if best_combo in self.region_rewards:
                best_avg = np.mean(self.region_rewards[best_combo])
                print(
                    f"Round {round_num}: {decision_type} (epsilon={current_epsilon:.3f}) - using best region (avg={best_avg:.4f})"
                )
            else:
                print(
                    f"Round {round_num}: {decision_type} (epsilon={current_epsilon:.3f}) - fallback to random"
                )

        # Generate actual hyperparameter values from selected regions
        suggested_params = self._sample_from_regions(selected_regions)

        # Store for update later
        self.last_suggested_regions = tuple(
            selected_regions[param] for param in sorted(selected_regions.keys())
        )

        print(f"Regions: {selected_regions}")
        print(f"Params: {suggested_params}")

        return suggested_params

    def _get_best_region_combination(self) -> Dict[str, int]:
        """Find the region combination with highest average reward"""
        if not self.region_rewards:
            # No data yet, return random
            return {
                param_name: random.choice(range(len(param_regions)))
                for param_name, param_regions in self.parameter_regions.items()
            }

        # Find combination with best average reward
        best_combo = None
        best_reward = -float("inf")

        for combo_tuple, rewards in self.region_rewards.items():
            if len(rewards) > 0:
                avg_reward = np.mean(rewards)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_combo = combo_tuple

        if best_combo is None:
            # Fallback to random
            return {
                param_name: random.choice(range(len(param_regions)))
                for param_name, param_regions in self.parameter_regions.items()
            }

        # Convert tuple back to dict
        param_names = sorted(self.parameter_regions.keys())
        return {param_names[i]: best_combo[i] for i in range(len(param_names))}

    def _sample_from_regions(self, selected_regions: Dict[str, int]) -> Dict:
        """
        Sample actual hyperparameter values from the selected regions
        """
        params = {}

        for param_name, region_idx in selected_regions.items():
            param_config = self.hyperparameter_space[param_name]
            region = self.parameter_regions[param_name][region_idx]

            if param_config["type"] == "int":
                # Sample random int from region
                min_val, max_val = region
                params[param_name] = random.randint(min_val, max_val)

            elif param_config["type"] == "float":
                # Sample random float from region
                min_val, max_val = region
                params[param_name] = random.uniform(min_val, max_val)

            elif param_config["type"] == "categorical":
                # Region contains the categorical value
                params[param_name] = region[0]

        return params

    def update(self, hyperparameters: Dict, score: float):
        """
        Update bandit with the result of the last suggestion
        """
        if self.last_suggested_regions is None:
            print("Warning: No region selection recorded for this update")
            return

        # Update region combination statistics
        combo_tuple = self.last_suggested_regions

        if combo_tuple not in self.region_rewards:
            self.region_rewards[combo_tuple] = []
            self.region_counts[combo_tuple] = 0

        self.region_rewards[combo_tuple].append(score)
        self.region_counts[combo_tuple] += 1

        # Update global best tracking with detailed logging
        old_best = self.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
            print(f"  *** NEW BEST SCORE: {score:.4f} (was {old_best:.4f}) ***")
        else:
            print(f"  No improvement: {score:.4f} vs best {self.best_score:.4f}")

        # Add to history
        self.history.append({"params": hyperparameters, "score": score})

        # Print learning progress
        avg_reward = np.mean(self.region_rewards[combo_tuple])
        print(
            f"Score: {score:.4f}, Region avg: {avg_reward:.4f} "
            f"(n={self.region_counts[combo_tuple]})"
        )

        # Debug: Show region learning
        print(f"  Total regions tried: {len(self.region_rewards)}")
        if len(self.region_rewards) > 1:
            region_avgs = [
                (combo, np.mean(rewards))
                for combo, rewards in self.region_rewards.items()
                if len(rewards) > 0
            ]
            region_avgs.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 3 regions by avg reward:")
            for i, (combo, avg) in enumerate(region_avgs[:3]):
                print(f"    {i+1}. Region {combo}: {avg:.4f}")

        # Reset for next round
        self.last_suggested_regions = None

    def get_region_statistics(self) -> Dict:
        """
        Get detailed statistics about region performance
        """
        stats = {
            "parameter_regions": self.parameter_regions,
            "region_rewards": {str(k): v for k, v in self.region_rewards.items()},
            "region_counts": {str(k): v for k, v in self.region_counts.items()},
            "best_regions": self._get_top_region_combinations(5),
        }
        return stats

    def _get_top_region_combinations(self, top_k: int = 5) -> List[Dict]:
        """Get the top-k performing region combinations"""
        region_performance = []

        for combo_tuple, rewards in self.region_rewards.items():
            if len(rewards) > 0:
                param_names = sorted(self.parameter_regions.keys())
                region_dict = {
                    param_names[i]: combo_tuple[i] for i in range(len(param_names))
                }

                region_performance.append(
                    {
                        "regions": region_dict,
                        "avg_reward": np.mean(rewards),
                        "count": len(rewards),
                        "std_reward": np.std(rewards) if len(rewards) > 1 else 0,
                    }
                )

        # Sort by average reward
        region_performance.sort(key=lambda x: x["avg_reward"], reverse=True)
        return region_performance[:top_k]
